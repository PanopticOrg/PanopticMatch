"""Points SIFT concordants entre deux images (RANSAC)."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

RATIO = 0.75
RANSAC_REPROJ = 5.0
MAX_SCALE = 8.0       # rapport d'échelle maximal entre les deux images
MAX_ANISOTROPY = 4.0  # au-delà, l'homographie écrase l'image sur une ligne ou un point
MIN_COVERAGE = 0.1    # part minimale de la page couverte par les points concordants, sur l'une des deux images
WORKERS = max(1, (os.cpu_count() or 4) // 4)  # laisse la machine utilisable pendant le calcul

cv2.setNumThreads(1)  # le parallélisme vient du pool : sans ça, OpenCV ouvre ses threads dans chaque worker

Features = tuple[np.ndarray, np.ndarray]  # (points (n, 2) float32, descripteurs (n, 128) float32)
Source = bytes | str  # image encodée, ou chemin du fichier


def extract(source: Source, nfeatures: int, max_size: int) -> Features | None:
    try:
        buf = np.frombuffer(source, np.uint8) if isinstance(source, bytes) else np.fromfile(source, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except (OSError, cv2.error):
        return None
    if img is None:
        return None
    scale = max_size / max(img.shape)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kps, desc = cv2.SIFT_create(nfeatures=nfeatures).detectAndCompute(img, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kps]), desc.astype(np.float32)


def count_inliers(a: Features, b: Features) -> int:
    (pts_a, desc_a), (pts_b, desc_b) = a, b
    if len(desc_a) < 4 or len(desc_b) < 4:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    # Un point de b ne peut répondre qu'à un seul point de a : sinon des motifs répétés (quadrillage,
    # écriture) se rabattent tous sur le même point et donnent une homographie dégénérée
    best: dict[int, cv2.DMatch] = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    if len(good) < 4:
        return 0
    src = pts_a[[m.queryIdx for m in good]]
    dst = pts_b[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if H is None or not _plausible(H):
        return 0
    inl = mask.ravel().astype(bool)
    # Un en-tête imprimé ou un tampon commun à deux lettres différentes ne couvre qu'un coin de la page
    if max(_coverage(src[inl], pts_a), _coverage(dst[inl], pts_b)) < MIN_COVERAGE:
        return 0
    # SIFT peut placer plusieurs points au même endroit (plusieurs orientations) : on compte les positions
    return min(len(np.unique(np.round(src[inl]), axis=0)), len(np.unique(np.round(dst[inl]), axis=0)))


def _coverage(inliers: np.ndarray, pts: np.ndarray) -> float:
    w, h = np.ptp(pts, axis=0)
    if len(inliers) < 3 or w * h == 0:
        return 0.0
    return cv2.contourArea(cv2.convexHull(inliers.astype(np.float32))) / (w * h)


def _plausible(H: np.ndarray) -> bool:
    if abs(H[2, 2]) < 1e-9 or not np.isfinite(H).all():
        return False
    sv = np.linalg.svd(H[:2, :2] / H[2, 2], compute_uv=False)
    return sv[1] > 0 and 1 / MAX_SCALE <= sv[1] <= sv[0] <= MAX_SCALE and sv[0] / sv[1] <= MAX_ANISOTROPY


class FeatureStore:
    """Points SIFT par sha1, gardés en mémoire et en cache disque."""

    def __init__(self, cache_dir: Path, nfeatures: int, max_size: int,
                 load_sources: Callable[[list[str]], dict[str, Source]]):
        self.cache_dir = cache_dir
        self.nfeatures = nfeatures
        self.max_size = max_size
        self.load_sources = load_sources
        self.features: dict[str, Features | None] = {}
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sha1: str) -> Path:
        return self.cache_dir / f"{sha1}.npz"

    def _read_cache(self, sha1: str) -> Features | None:
        try:
            with np.load(self._path(sha1)) as f:
                return f["pts"], f["desc"].astype(np.float32)
        except (OSError, KeyError, ValueError):
            return None

    def _compute(self, sha1: str, source: Source) -> tuple[str, Features | None]:
        feats = extract(source, self.nfeatures, self.max_size)
        if feats is not None:
            tmp = self._path(sha1).with_suffix(".tmp.npz")
            np.savez(tmp, pts=feats[0], desc=feats[1].astype(np.float16))
            os.replace(tmp, self._path(sha1))
        return sha1, feats

    def ensure(self, sha1s: Iterable[str]) -> None:
        missing = []
        for sha1 in dict.fromkeys(sha1s):
            if sha1 in self.features:
                continue
            cached = self._read_cache(sha1)
            if cached is None:
                missing.append(sha1)
            else:
                self.features[sha1] = cached
        if not missing:
            return
        data = self.load_sources(missing)
        for sha1 in missing:
            if sha1 not in data:
                self.features[sha1] = None
        with ThreadPoolExecutor(WORKERS) as pool:
            for sha1, feats in pool.map(lambda s: self._compute(s, data[s]), [s for s in missing if s in data]):
                self.features[sha1] = feats

    def inliers(self, pairs: list[tuple[str, str]]) -> list[int]:
        def one(pair):
            a, b = self.features.get(pair[0]), self.features.get(pair[1])
            return count_inliers(a, b) if a is not None and b is not None else 0

        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(one, pairs))
