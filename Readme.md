# Plugin EMAN pour Panoptic

Ce Plugin ajoute les fonctions suivantes à panoptic:
- **regrouper_en_documents:** grouper des images (étant dans un ordre séquentiel) en fonction de leurs largeur et hauteur, avec une marge d'erreur de 3% par défaut, mais ce paramètre peut être spécifié au moment de lancer la fonction. Le résultat de ce groupage est sauvegardé sous la forme d'une propriété "group_number". On peut venir ensuite modifier cette propriété sur les images qu'on ne veut plus avoir dans un même groupe.
- **trouver_image_originale:** fonction qui prend en paramètre deux dossiers et trouve par similarité visuelle pour chaque image du dossier A l'image du dossier B qui lui est la plus proche. Ces deux dossiers doivent être rentrés en paramètre lors de l'execution de la fonction AINSI qu'un dossier racine si jamais ces dossiers n'ont pas été ajoutés individuellement à Panoptic.

## Installation

Sur la page d'accueil de Panoptic, ajoutez un nouveau plugin de type git en mettant https://gitlab.com/eman8/panopticeman en url. Normalement cela devrait marcher tout seul.

## Utilisation

Une fois installé les fonctions sont disponibles en cliquant sur le bouton "executer" (représenté potentiellement par un petit logo de terminal en fonction de la version de Panoptic).

## Exemples:

### Trouver Image Originale

Pour bien comprendre les dossiers, j'ai ajouté un dossier d'images à mon panoptic. 

Ce dossier est situé à `/home/ceres/Images/corpus/Flam` et contient les dossiers suivants:
```
└── Flam
    ├── Nouvelles
    └── Anciennes
```

Si je veux pouvoir trouver toutes les images similaires aux images de Nouvelles dans le dossier Anciennes, je dois executer la fonction avec les paramètres suivants:

- root_folder: `/home/ceres/Images/corpus/Flam` 
- input_folder: `Nouvelles`
- comparison_folder: `Anciennes`
