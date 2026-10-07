"""
DocumentSegmenterPlugin
=======================
Segmente une séquence d'images numérisées (triées par filepath) en groupes
cohérents ("documents"), en s'appuyant sur des vecteurs existants.

Contraintes strictes :
  - L'ordre des images EST respecté (pas de groupe "non-contigu")
  - Chaque groupe comporte entre 1 et MAX_GROUP_SIZE images (défaut 10)
  - Un algorithme de programmation dynamique minimise la dissimilarité
    cosine intra-groupe sur la séquence ordonnée
  - La propriété "group_number" (type number) est créée si absente,
    puis renseignée pour chaque instance traitée
"""
# No `from __future__ import annotations`: the action registry reads the real annotation classes.

import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from panoptic.core.databases.data.models import InstanceValue, UpsertCommit
from panoptic.core.databases.entity_schema import OP_UPDATE
from panoptic.core.databases.media.models import VectorType
from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, FolderPath, Group, Notif, NotifType, Score,
)
from panoptic.models.models import PropertyMode, PropertyType


# ---------------------------------------------------------------------------
# Paramètres configurables via l'UI Panoptic
# ---------------------------------------------------------------------------

class SegmenterParams(BaseModel):
    max_group_size: int = 10


# ---------------------------------------------------------------------------
# Plugin principal
# ---------------------------------------------------------------------------

class DocumentSegmenterPlugin(APlugin):
    """
    Segmente des images numérisées ordonnées en groupes de documents cohérents.
    """

    def __init__(self, name: str, project, plugin_path: str):
        self.params = SegmenterParams()
        super().__init__(name=name, project=project, plugin_path=plugin_path)

        self.add_action_easy(self.regrouper_en_documents, ["execute"])
        self.add_action_easy(self.trouver_image_originale, ["execute"])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def regrouper_en_documents(self, ctx: ActionContext, pourcentage: int = 3) -> ActionResult:
        """Regroupe les images consécutives tant que leurs dimensions varient de moins de `pourcentage` %.
        @pourcentage: variation maximale (en %) de largeur ou de hauteur entre deux images d'un même groupe
        """
        images = sorted(self._get_images(ctx), key=lambda img: img.path)
        if not images:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Aucune image", "Aucune image à segmenter.")])

        group_assignments = _segment_by_dims(images, pourcentage / 100)

        group_number_prop = self._get_or_create_group_number_property()
        commit = UpsertCommit()
        commit.instance_values[group_number_prop.id] = [
            InstanceValue(
                property_id=group_number_prop.id,
                instance_id=images[idx].instance.id,
                value=group_idx + 1,
                commit_id=0,
                operation=OP_UPDATE,
            )
            for group_idx, indices in enumerate(group_assignments)
            for idx in indices
        ]
        self.project.apply_upsert_commit(commit)

        result_groups = []
        for group_idx, indices in enumerate(group_assignments):
            ids = [images[i].instance.id for i in indices]
            first_file = os.path.basename(images[indices[0]].path)
            last_file = os.path.basename(images[indices[-1]].path)
            result_groups.append(Group(ids=ids, name=f"Doc {group_idx + 1}  [{first_file} … {last_file}]"))

        return ActionResult(
            groups=result_groups,
            notifs=[Notif(NotifType.INFO, "Segmentation terminée",
                          f"{len(images)} images en {len(result_groups)} groupes.")]
        )

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_images(self, ctx: ActionContext) -> list["_Image"]:
        """Instances du contexte avec le chemin complet et les dimensions de leur fichier."""
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
            images.append(_Image(inst, path, file.width, file.height))
        return images

    def _get_or_create_group_number_property(self):
        """
        Retourne la propriété "group_number" (type number, mode instance).
        La crée si elle n'existe pas encore.
        """
        prop_name = "group_number"
        for prop in self.project.get_properties(name=prop_name):
            if prop.mode == PropertyMode.id.value:
                return prop
        return self.project.add_property(prop_name, PropertyType.number.value, mode=PropertyMode.id.value)


class _Image:
    __slots__ = ("instance", "path", "width", "height")

    def __init__(self, instance, path: str, width: int | None, height: int | None):
        self.instance = instance
        self.path = path
        self.width = width
        self.height = height


def _to_path(p: str) -> Path:
    return Path(p.replace("\\", "/"))


def _segment_by_dims(images, percentage: float = 0.03):
    """Coupe la séquence en groupes dès qu'un changement de dimensions > 3% est détecté."""
    groups = []
    current = [0]

    for i in range(1, len(images)):
        prev = images[i - 1]
        curr = images[i]

        if prev.width and prev.height and curr.width and curr.height:
            w_ratio = abs(prev.width - curr.width) / max(prev.width, curr.width)
            h_ratio = abs(prev.height - curr.height) / max(prev.height, curr.height)
            if w_ratio > percentage or h_ratio > percentage:
                groups.append(current)
                current = []

        current.append(i)

    groups.append(current)
    return groups
