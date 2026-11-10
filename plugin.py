"""
ImageMatchPlugin
================
Pour chaque image d'un dossier, trouve l'image correspondante d'un autre dossier :
les vecteurs existants présélectionnent les candidats, les points SIFT concordants décident.
"""
# No `from __future__ import annotations`: the action registry reads the real annotation classes.

import os
from collections import Counter
from pathlib import Path

import numpy as np

from panoptic.core.databases.media.models import VectorType
from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, FolderPath, Group, Notif, NotifType, Score,
)

from .sift import FeatureStore

SIFT_IMAGE_SIZE = 1024


class ImageMatchPlugin(APlugin):
    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.add_action_easy(self.trouver_image_originale, ["execute"])

    # Defaults are needed: an input left empty in the UI is not passed to the function.
    def trouver_image_originale(self, ctx: ActionContext, vec_type: VectorType = None,
                                input_folder: FolderPath = None,
                                comparison_folder: FolderPath = None,
                                top_k: int = 100,
                                min_inliers: int = 20,
                                max_matches: int = 1,
                                sift_nfeatures: int = 1000) -> ActionResult:
        """Pour chaque image du dossier d'entrée, trouve l'image correspondante du dossier de comparaison.
        @vec_type: vecteurs utilisés pour présélectionner les candidats
        @input_folder: dossier contenant les images dont on cherche l'original
        @comparison_folder: dossier dans lequel chercher les originaux
        @top_k: nombre de candidats comparés par points SIFT pour chaque image d'entrée
        @min_inliers: nombre minimal de points concordants pour garder une paire
        @max_matches: nombre maximal de paires par image, par exemple 2 pour des doubles pages découpées en deux fichiers
        @sift_nfeatures: nombre de points SIFT extraits par image
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

        groups = self._match_groups(input_images, comparison_images, input_vecs, comparison_vecs, sims,
                                    top_k, min_inliers, max(1, max_matches), sift_nfeatures)

        return ActionResult(
            groups=groups,
            notifs=[Notif(NotifType.INFO, "Matching terminé",
                          f"{len(groups)} correspondances trouvées (>= {min_inliers} points concordants) "
                          f"sur {len(input_images)} images d'entrée.")]
        )

    def _match_groups(self, input_images, comparison_images, input_vecs, comparison_vecs, sims,
                      top_k, min_inliers, max_matches, sift_nfeatures) -> list[Group]:
        # Le centrage retire la composante commune au corpus (même type de document) pour classer les candidats
        mean = np.concatenate([input_vecs, comparison_vecs]).mean(axis=0)
        ranking = _normalize(input_vecs - mean) @ _normalize(comparison_vecs - mean).T
        k = max(1, min(top_k, ranking.shape[1]))
        cand = np.argpartition(-ranking, k - 1, axis=1)[:, :k]

        pairs = [(i, int(j)) for i in range(len(input_images)) for j in cand[i]]

        image_type = self._sift_image_type()
        if image_type is None:
            paths = {img.instance.sha1: img.path for img in input_images + comparison_images}
            source, load = "file", lambda sha1s: {s: paths[s] for s in sha1s if s in paths}
        else:
            source, load = f"type{image_type}", lambda sha1s: self._load_image_bytes(image_type, sha1s)
        store = FeatureStore(self.data_path / f"sift_{source}_{SIFT_IMAGE_SIZE}_{sift_nfeatures}",
                             sift_nfeatures, SIFT_IMAGE_SIZE, load)
        store.ensure([input_images[i].instance.sha1 for i, _ in pairs] +
                     [comparison_images[j].instance.sha1 for _, j in pairs])
        inliers = store.inliers([(input_images[i].instance.sha1, comparison_images[j].instance.sha1)
                                 for i, j in pairs])

        scored = sorted(((n, float(sims[i, j]), i, j) for n, (i, j) in zip(inliers, pairs) if n >= min_inliers),
                        reverse=True)
        # Appariement glouton : les paires les plus sûres d'abord, chaque image dans au plus max_matches paires
        input_count, comparison_count, kept = Counter(), Counter(), []
        for n, sim, i, j in scored:
            if input_count[i] >= max_matches or comparison_count[j] >= max_matches:
                continue
            input_count[i] += 1
            comparison_count[j] += 1
            kept.append((n, i, j))
        return _connected_groups(input_images, comparison_images, kept)

    def _sift_image_type(self) -> int | None:
        """Plus petit rendu stocké d'au moins SIFT_IMAGE_SIZE px, sinon None : les fichiers originaux sont lus."""
        with self.project._media_db() as db:
            types = db.get_image_types()
        big_enough = [(max(t.width or 0, t.height or 0), t.id) for t in types
                      if max(t.width or 0, t.height or 0) >= SIFT_IMAGE_SIZE]
        return min(big_enough)[1] if big_enough else None

    def _load_image_bytes(self, image_type: int, sha1s: list[str]) -> dict[str, bytes]:
        data = {}
        for start in range(0, len(sha1s), 200):
            with self.project._media_db() as db:
                for img in db.get_images(type_id=image_type, sha1=sha1s[start:start + 200]):
                    data[img.sha1] = img.data
        return data

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


def _connected_groups(input_images, comparison_images, kept: list[tuple[int, int, int]]) -> list[Group]:
    """Un groupe par ensemble de paires reliées : une image ne peut appartenir qu'à un seul groupe dans Panoptic.
    Avec max_matches > 1, une double page et ses deux moitiés forment ainsi un seul groupe."""
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in kept:
        parent[find(("in", i))] = find(("cmp", j))
    components: dict = {}
    for n, i, j in kept:
        components.setdefault(find(("in", i)), []).append((n, i, j))

    best = max((n for n, _, _ in kept), default=0)
    groups = []
    for pairs in components.values():
        inputs = list(dict.fromkeys(input_images[i] for _, i, _ in pairs))
        matches = list(dict.fromkeys(comparison_images[j] for _, _, j in pairs))
        weakest = min(n for n, _, _ in pairs)
        groups.append(Group(
            ids=[img.instance.id for img in inputs + matches],
            name=f"{', '.join(os.path.basename(img.path) for img in inputs)}  ↔  "
                 f"{', '.join(os.path.basename(img.path) for img in matches)}",
            score=Score(value=weakest, min=0, max=best, max_is_best=True,
                        description="Points SIFT concordants (paire la plus faible du groupe)")))
    groups.sort(key=lambda g: -g.score.value)
    return groups


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class _Image:
    __slots__ = ("instance", "path")

    def __init__(self, instance, path: str):
        self.instance = instance
        self.path = path


def _to_path(p: str) -> Path:
    return Path(p.replace("\\", "/"))
