"""
ImageMatchPlugin
================
Pour chaque image d'un dossier, trouve l'image la plus similaire d'un autre dossier,
en s'appuyant sur des vecteurs existants.
"""
# No `from __future__ import annotations`: the action registry reads the real annotation classes.

import os
from pathlib import Path

import numpy as np

from panoptic.core.databases.media.models import VectorType
from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, FolderPath, Group, Notif, NotifType, Score,
)


class ImageMatchPlugin(APlugin):
    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.add_action_easy(self.trouver_image_originale, ["execute"])

    # Defaults are needed: an input left empty in the UI is not passed to the function.
    def trouver_image_originale(self, ctx: ActionContext, vec_type: VectorType = None,
                                input_folder: FolderPath = None,
                                comparison_folder: FolderPath = None,
                                similarity_threshold: float = 0.8) -> ActionResult:
        """Pour chaque image du dossier d'entrée, trouve l'image la plus similaire du dossier de comparaison.
        @vec_type: vecteurs utilisés pour calculer la similarité
        @input_folder: dossier contenant les images dont on cherche l'original
        @comparison_folder: dossier dans lequel chercher les originaux
        @similarity_threshold: similarité cosine minimale (entre 0 et 1) pour garder une paire
        """
        if vec_type is None or not input_folder or not comparison_folder:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Paramètres manquants",
                                              "Choisir un type de vecteur, un dossier d'entrée et un dossier de comparaison.")])

        input_path = _to_path(input_folder)
        comparison_path = _to_path(comparison_folder)
        images = self._get_images(ctx)

        input_images = [img for img in images if _to_path(img.path).is_relative_to(input_path)]
        input_ids = {img.instance.id for img in input_images}
        comparison_images = [img for img in images
                             if _to_path(img.path).is_relative_to(comparison_path) and img.instance.id not in input_ids]

        if not input_images:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Dossier vide",
                                              f"Aucune image trouvée dans le dossier d'entrée : {input_folder}")])
        if not comparison_images:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Dossier vide",
                                              f"Aucune image trouvée dans le dossier de comparaison : {comparison_folder}")])

        # Récupérer tous les vecteurs
        all_sha1s = list({img.instance.sha1 for img in input_images + comparison_images})
        vectors_raw = self.project.get_vectors(vec_type.id, sha1s=all_sha1s)

        vec_map: dict[str, np.ndarray] = {}
        for v in vectors_raw:
            arr = np.asarray(v.data, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vec_map[v.sha1] = arr

        input_images = sorted((img for img in input_images if img.instance.sha1 in vec_map), key=lambda i: i.path)
        comparison_images = [img for img in comparison_images if img.instance.sha1 in vec_map]
        if not input_images or not comparison_images:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Pas de vecteurs",
                                              "Aucun vecteur trouvé pour les images d'un des deux dossiers.")])

        # Similarité cosine entre chaque image d'entrée (lignes) et chaque image de comparaison (colonnes)
        input_vecs = np.stack([vec_map[img.instance.sha1] for img in input_images])
        comparison_vecs = np.stack([vec_map[img.instance.sha1] for img in comparison_images])
        sims = input_vecs @ comparison_vecs.T

        # Appariement glouton global : les paires les plus similaires sont attribuées en premier,
        # chaque image ne pouvant appartenir qu'à une seule paire
        rows, cols = np.nonzero(sims >= similarity_threshold)
        order = np.argsort(-sims[rows, cols], kind="stable")
        used_inputs, used_comparisons = set(), set()
        groups = []
        for k in order:
            i, j = int(rows[k]), int(cols[k])
            if i in used_inputs or j in used_comparisons:
                continue
            used_inputs.add(i)
            used_comparisons.add(j)

            inp, match = input_images[i], comparison_images[j]
            sim = round(float(sims[i, j]) * 100, 1)
            groups.append(Group(
                ids=[inp.instance.id, match.instance.id],
                name=f"{os.path.basename(inp.path)}  ↔  {os.path.basename(match.path)}",
                score=Score(value=sim, min=0, max=100, max_is_best=True, description="Similarité cosine")
            ))

        return ActionResult(
            groups=groups,
            notifs=[Notif(NotifType.INFO, "Matching terminé",
                          f"{len(groups)} paires trouvées (similarité >= {similarity_threshold}) "
                          f"sur {len(input_images)} images d'entrée.")]
        )

    def _get_images(self, ctx: ActionContext) -> list["_Image"]:
        """Instances du contexte avec le chemin complet de leur fichier."""
        if ctx.instance_ids:
            instances = self.project.get_instances(id=ctx.instance_ids)
        else:
            instances = self.project.get_instances()
        if not instances:
            return []

        files = {f.id: f for f in self.project.get_files(id=list({i.file_id for i in instances}))}
        folders = {f.id: f for f in self.project.get_folders(id=list({f.folder_id for f in files.values()}))}

        images = []
        for inst in instances:
            file = files.get(inst.file_id)
            if file is None or not file.name:
                continue
            folder = folders.get(file.folder_id)
            path = os.path.join(folder.path, file.name) if folder and folder.path else file.name
            images.append(_Image(inst, path))
        return images


class _Image:
    __slots__ = ("instance", "path")

    def __init__(self, instance, path: str):
        self.instance = instance
        self.path = path


def _to_path(p: str) -> Path:
    return Path(p.replace("\\", "/"))
