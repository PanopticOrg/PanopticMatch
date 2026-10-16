# Plugin MATCH pour Panoptic

Ajoute la fonction **trouver_image_originale** : pour chaque image d'un dossier d'entrée, trouve par similarité visuelle l'image la plus proche d'un dossier de comparaison. Chaque image n'apparaît que dans une seule paire, les paires sous le seuil sont ignorées et le résultat est trié par similarité décroissante.

Nécessite Panoptic >= 1.0.0 et des vecteurs déjà calculés (par exemple avec PanopticML).

## Installation

Sur la page d'accueil de Panoptic, ajoutez un plugin de type git avec l'url https://github.com/PanopticOrg/PanopticMatch.

## Utilisation

Lancez `trouver_image_originale` depuis le bouton « exécuter » avec :

- `vec_type` : type de vecteurs à utiliser ;
- `input_folder` : dossier des images dont on cherche l'original (ex. `.../Flam/Nouvelles`) ;
- `comparison_folder` : dossier où chercher les originaux (ex. `.../Flam/Anciennes`) ;
- `similarity_threshold` : similarité cosine minimale, `0.8` par défaut.
