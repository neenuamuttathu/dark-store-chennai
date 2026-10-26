"""Synthetic Chennai instance: customer demand points and candidate dark-store sites."""

import numpy as np
import pandas as pd

# Approximate centres of residential localities (lat, lon). Most customers cluster
# around these; the rest are spread uniformly over the city.
LOCALITIES = {
    "T. Nagar": (13.0418, 80.2341),
    "Anna Nagar": (13.0850, 80.2101),
    "Adyar": (13.0012, 80.2565),
    "Velachery": (12.9815, 80.2180),
    "Mylapore": (13.0368, 80.2676),
    "Tambaram": (12.9249, 80.1170),
    "Porur": (13.0382, 80.1565),
    "Guindy": (13.0067, 80.2206),
    "Ambattur": (13.1143, 80.1548),
    "Perambur": (13.1210, 80.2330),
    "Sholinganallur": (12.9010, 80.2279),
    "Chromepet": (12.9516, 80.1462),
    "Kodambakkam": (13.0521, 80.2255),
    "Thiruvanmiyur": (12.9830, 80.2560),
    "Vadapalani": (13.0500, 80.2121),
    "Pallavaram": (12.9675, 80.1491),
    "Madipakkam": (12.9623, 80.1986),
    "Kolathur": (13.1240, 80.2120),
    "Medavakkam": (12.9200, 80.1920),
    "Thiruvottiyur": (13.1600, 80.2900),
    "Royapuram": (13.1100, 80.2850),
    "Egmore": (13.0732, 80.2609),
    "Nungambakkam": (13.0569, 80.2425),
    "Valasaravakkam": (13.0400, 80.1750),
    "Perungudi": (12.9654, 80.2461),
    "Mogappair": (13.0837, 80.1750),
    "Saidapet": (13.0213, 80.2231),
    "Pallikaranai": (12.9349, 80.2137),
}

CITY_CENTRE = (13.0604, 80.2496)  # Teynampet / Gemini flyover area
LAT_MIN, LAT_MAX = 12.86, 13.20
LON_MIN = 80.09

# Coastline of the Bay of Bengal, south to north (lat, lon). Land lies west of it.
COAST = np.array([
    (12.85, 80.247), (12.90, 80.252), (12.95, 80.258), (12.98, 80.265),
    (13.00, 80.272), (13.05, 80.283), (13.08, 80.290), (13.11, 80.296),
    (13.16, 80.305), (13.20, 80.315),
])
COAST_BUFFER_DEG = 0.004  # keep points ~400 m inland

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(13.05))


def on_land(lat, lon):
    coast_lon = np.interp(lat, COAST[:, 0], COAST[:, 1]) - COAST_BUFFER_DEG
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= coast_lon)


def _sample_points(rng, n, cluster_share, sigma_km):
    """Mixture of Gaussian clusters around localities and a uniform background, clipped to land."""
    centres = np.array(list(LOCALITIES.values()))
    points = np.empty((0, 2))
    while len(points) < n:
        m = 2 * (n - len(points)) + 100
        clustered = rng.random(m) < cluster_share
        k = rng.integers(len(centres), size=m)
        lat = np.where(
            clustered,
            centres[k, 0] + rng.normal(0, sigma_km / KM_PER_DEG_LAT, m),
            rng.uniform(LAT_MIN, LAT_MAX, m),
        )
        lon = np.where(
            clustered,
            centres[k, 1] + rng.normal(0, sigma_km / KM_PER_DEG_LON, m),
            rng.uniform(LON_MIN, COAST[:, 1].max(), m),
        )
        keep = on_land(lat, lon)
        points = np.vstack([points, np.column_stack([lat[keep], lon[keep]])])
    return points[:n]


def _nearest_locality(lat, lon):
    names = list(LOCALITIES)
    centres = np.array(list(LOCALITIES.values()))
    dy = (lat[:, None] - centres[None, :, 0]) * KM_PER_DEG_LAT
    dx = (lon[:, None] - centres[None, :, 1]) * KM_PER_DEG_LON
    return [names[k] for k in np.argmin(dx**2 + dy**2, axis=1)]


def generate_instance(n_customers, n_candidates, seed=42):
    """Return (customers, candidates) DataFrames.

    Candidates and customers use independent random streams, so changing the number
    of customers does not move the candidate sites (and vice versa).
    """
    rng_sites = np.random.default_rng([seed, 1])
    rng_cust = np.random.default_rng([seed, 2])

    pts = _sample_points(rng_cust, n_customers, cluster_share=0.75, sigma_km=1.2)
    customers = pd.DataFrame({
        "customer": [f"C{i + 1:05d}" for i in range(n_customers)],
        "lat": pts[:, 0],
        "lon": pts[:, 1],
        "orders": rng_cust.integers(2, 11, n_customers),  # orders per month, 2-10
    })

    pts = _sample_points(rng_sites, n_candidates, cluster_share=0.8, sigma_km=1.5)
    dy = (pts[:, 0] - CITY_CENTRE[0]) * KM_PER_DEG_LAT
    dx = (pts[:, 1] - CITY_CENTRE[1]) * KM_PER_DEG_LON
    centrality = np.exp(-np.hypot(dx, dy) / 8.0)
    # 0 = cheapest, 1 = most expensive site; scaled to the fixed-cost range in the model.
    rent_index = np.clip(0.7 * centrality + 0.3 * rng_sites.random(n_candidates), 0, 1)
    candidates = pd.DataFrame({
        "store": [f"S{j + 1:02d}" for j in range(n_candidates)],
        "lat": pts[:, 0],
        "lon": pts[:, 1],
        "locality": _nearest_locality(pts[:, 0], pts[:, 1]),
        "rent_index": rent_index,
    })
    return customers, candidates
