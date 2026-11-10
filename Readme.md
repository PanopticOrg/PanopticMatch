# Plugin MATCH pour Panoptic

Ajoute la fonction **trouver_image_originale** : pour chaque image d'un dossier d'entrée, trouve l'image correspondante d'un dossier de comparaison (par exemple l'original d'une image retouchée ou recadrée). Par défaut, chaque image n'apparaît que dans une seule paire.

Les vecteurs présélectionnent les candidats les plus proches, puis chaque candidat est comparé en comptant les points SIFT concordants entre les deux images (RANSAC), comme PIMMI. Ce comptage évite les faux appariements entre images de même forme mais de contenu différent (pages manuscrites, documents d'un même fonds…), que la similarité cosine seule ne sait pas distinguer. Les paires sont triées par nombre de points concordants décroissant.

Nécessite Panoptic >= 1.0.0 et des vecteurs déjà calculés (par exemple avec PanopticML).

## Installation

Sur la page d'accueil de Panoptic, ajoutez un plugin de type git avec l'url https://github.com/PanopticOrg/PanopticMatch.

## Utilisation

Lancez `trouver_image_originale` depuis le bouton « exécuter » avec :

- `vec_type` : type de vecteurs à utiliser ;
- `input_folder` : images à apparier ;
- `comparison_folder` : images candidates ;
- `top_k` : nombre de candidats comparés par image d'entrée, `100` par défaut, à augmenter si des originaux sont manqués ;
- `min_inliers` : nombre minimal de points concordants pour garder une paire, `20` par défaut, à augmenter pour des paires plus sûres ;
- `max_matches` : nombre maximal de paires par image, `1` par défaut ; mettre `2` quand des doubles pages ont été découpées en deux fichiers, pour que la double page soit appariée à ses deux moitiés ;
- `sift_nfeatures` : nombre de points SIFT extraits par image, `1000` par défaut.

Les points SIFT sont calculés sur les miniatures de 1024 px stockées par Panoptic et gardés en cache dans les données du plugin : les lancements suivants sont bien plus rapides.
