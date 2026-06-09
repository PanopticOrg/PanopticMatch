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

import os
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, DbCommit, InstanceProperty, PropertyId
from panoptic.models.results import ActionResult, Group, Notif, NotifType, Score
from panoptic.models import PropertyType, PropertyMode, VectorType


# ---------------------------------------------------------------------------
# Paramètres configurables via l'UI Panoptic
# ---------------------------------------------------------------------------

class SegmenterParams(BaseModel):
    max_group_size: int = 10
    similarity_threshold: float = 0.20


# ---------------------------------------------------------------------------
# Plugin principal
# ---------------------------------------------------------------------------

class DocumentSegmenterPlugin(APlugin):
    """
    Segmente des images numérisées ordonnées en groupes de documents cohérents.
    """

    def __init__(self, project, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params: SegmenterParams = SegmenterParams()

        self.add_action_easy(self.regrouper_en_documents, ["execute"])
        self.add_action_easy(self.trouver_image_originale, ["execute"])

    # ------------------------------------------------------------------
    # Action principale
    # ------------------------------------------------------------------

    async def regrouper_en_documents(self, ctx: ActionContext, pourcentage: int = 3) -> ActionResult:
        instances = await self.project.get_instances(ids=ctx.instance_ids)
        instances_sorted = sorted(instances, key=lambda inst: inst.url or "")

        group_assignments = _segment_by_dims(instances_sorted, pourcentage / 100)

        group_number_prop = await _get_or_create_group_number_property(self.project)
        commit = DbCommit()
        commit.instance_values = []
        for group_idx, group_instance_indices in enumerate(group_assignments):
            for idx in group_instance_indices:
                commit.instance_values.append(InstanceProperty(
                    instance_id=instances_sorted[idx].id,
                    property_id=group_number_prop.id,
                    value=group_idx + 1
                ))
        await self.project.do(commit)

        result_groups = []
        for group_idx, indices in enumerate(group_assignments):
            ids = [instances_sorted[i].id for i in indices]
            first_file = os.path.basename(instances_sorted[indices[0]].url or "")
            last_file = os.path.basename(instances_sorted[indices[-1]].url or "")
            result_groups.append(Group(ids=ids, name=f"Doc {group_idx + 1}  [{first_file} … {last_file}]"))

        return ActionResult(
            groups=result_groups,
            notifs=[Notif(NotifType.INFO, "Segmentation terminée",
                          f"{len(instances_sorted)} images en {len(result_groups)} groupes.")]
        )

    async def trouver_image_originale(self, ctx: ActionContext, vec_type: VectorType, root_folder: str, input_folder: str, comparison_folder: str) -> ActionResult:
        all_instances = await self.project.get_instances(ids=ctx.instance_ids)
        def _to_path(p: str) -> Path:
            return Path(p.replace("\\", "/"))

        input_path = _to_path(root_folder) / _to_path(input_folder)
        comparison_path = _to_path(root_folder) / _to_path(comparison_folder)

        input_instances = [i for i in all_instances if i.url and _to_path(i.url).is_relative_to(input_path)]
        comparison_instances = [i for i in all_instances if i.url and _to_path(i.url).is_relative_to(comparison_path)]

        if not input_instances:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Dossier vide",
                                              f"Aucune image trouvée dans le dossier d'entrée : {input_folder}")])
        if not comparison_instances:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Dossier vide",
                                              f"Aucune image trouvée dans le dossier de comparaison : {comparison_folder}")])

        # Récupérer tous les vecteurs
        all_sha1s = list({i.sha1 for i in input_instances + comparison_instances})
        vectors_raw = await self.project.get_vectors(vec_type.id, sha1s=all_sha1s)

        vec_map: dict[str, np.ndarray] = {}
        for v in vectors_raw:
            arr = np.array(v.data, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            vec_map[v.sha1] = arr

        # Pour chaque image d'entrée, trouver la plus proche dans le dossier de comparaison
        comparison_sha1s = [i.sha1 for i in comparison_instances]
        comparison_vecs = np.array([vec_map[s] for s in comparison_sha1s if s in vec_map], dtype=np.float32)
        comparison_sha1s_filtered = [s for s in comparison_sha1s if s in vec_map]

        if len(comparison_vecs) == 0:
            return ActionResult(notifs=[Notif(NotifType.ERROR, "Pas de vecteurs",
                                              "Aucun vecteur trouvé pour les images du dossier de comparaison.")])

        groups = []
        used_comparison_sha1s = set()


        for inst in sorted(input_instances, key=lambda i: i.url or ""):
            if inst.sha1 not in vec_map:
                continue

            input_vec = vec_map[inst.sha1]
            # Similarité cosine avec toutes les images de comparaison
            sims = comparison_vecs @ input_vec

            # Trouver le meilleur match non encore utilisé
            sorted_indices = np.argsort(sims)[::-1]
            best_sha1 = None
            for idx in sorted_indices:
                candidate = comparison_sha1s_filtered[idx]
                if candidate not in used_comparison_sha1s:
                    best_sha1 = candidate
                    break

            if best_sha1 is None:
                continue

            used_comparison_sha1s.add(best_sha1)
            input_filename = os.path.basename(inst.url or "")
            match_url = next((i.url for i in comparison_instances if i.sha1 == best_sha1), "")
            match_filename = os.path.basename(match_url)

            best_idx = comparison_sha1s_filtered.index(best_sha1)
            best_sim = round(float(sims[best_idx]), 3)

            groups.append(Group(
                sha1s=[inst.sha1, best_sha1],
                name=f"{input_filename}  ↔  {match_filename}",
                score=Score(value=best_sim * 100, min=0, max=100, max_is_best=True, description="Similarité cosine")
            ))

        return ActionResult(
            groups=groups,
            notifs=[Notif(NotifType.INFO, "Matching terminé",
                          f"{len(groups)} paires trouvées sur {len(input_instances)} images d'entrée.")]
        )

def _segment_by_dims(instances, percentage: float = 0.03):
    """Coupe la séquence en groupes dès qu'un changement de dimensions > 3% est détecté."""
    groups = []
    current = [0]

    for i in range(1, len(instances)):
        prev = instances[i - 1]
        curr = instances[i]

        if prev.width and prev.height and curr.width and curr.height:
            w_ratio = abs(prev.width - curr.width) / max(prev.width, curr.width)
            h_ratio = abs(prev.height - curr.height) / max(prev.height, curr.height)
            if w_ratio > percentage or h_ratio > percentage:
                groups.append(current)
                current = []

        current.append(i)

    groups.append(current)
    return groups

# ---------------------------------------------------------------------------
# Helper : propriété "group_number"
# ---------------------------------------------------------------------------

async def _get_or_create_group_number_property(project):
    """
    Retourne la propriété "group_number" (type number, mode instance).
    La crée via DbCommit si elle n'existe pas encore.
    """
    PROP_NAME = "group_number"

    all_props = await project.get_properties()
    for prop in all_props:
        if prop.name == PROP_NAME:
            return prop

    # Créer la propriété
    commit = DbCommit()
    new_prop = project.create_property(
        name=PROP_NAME,
        type_=PropertyType.number,
        mode=PropertyMode.id
    )
    commit.properties = [new_prop]
    result_commit = await project.do(commit)

    # Récupérer la propriété avec son vrai ID
    all_props = await project.get_properties()
    for prop in all_props:
        if prop.name == PROP_NAME:
            return prop

    raise RuntimeError("Impossible de créer la propriété 'group_number'.")
