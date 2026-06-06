from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from repd.constants import (
    CANCELLED_DEVELOPMENT_TYPES,
    DATETIME_COLS,
    SUCCESSFUL_DEVELOPMENT_TYPES,
    UNSUCCESSFUL_DEVELOPMENT_TYPES,
)

_DEFAULT_SRC = Path(__file__).parent.parent / "data" / "REPD_Publication_Q4_2025.csv"


class REPDProcessor:
    """Load, filter, and transform a REPD quarterly CSV into a GeoDataFrame."""

    def __init__(
        self,
        src: str | Path = _DEFAULT_SRC,
        encoding: str = "utf-8",
    ) -> None:
        self.src = Path(src)
        self.encoding = encoding

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Read the raw REPD CSV into a DataFrame."""
        return pd.read_csv(self.src, encoding=self.encoding)

    # ── Transforms ───────────────────────────────────────────────────────────

    def convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse all date columns to ``datetime64``.

        Uses ``dayfirst=True`` to match the REPD ``dd/mm/yyyy`` format.
        """
        for col in DATETIME_COLS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        return df

    def coordinates_to_lat_lon(
        self,
        df: pd.DataFrame,
        drop_source_cols: bool = True,
        easting_col: str = "X-coordinate",
        northing_col: str = "Y-coordinate",
        from_crs: str = "EPSG:27700",
        to_crs: str = "EPSG:4326",
    ) -> pd.DataFrame:
        """Convert British National Grid (OSGB36) coordinates to WGS84 lat/lon.

        Rows with missing or non-numeric coordinates are dropped. The resulting
        ``lat`` and ``lon`` columns are rounded to 4 decimal places (~11 m
        precision).

        Args:
            df: DataFrame containing REPD easting/northing columns.
            drop_source_cols: Remove the original easting/northing columns.
            easting_col: Name of the easting column in the source data.
            northing_col: Name of the northing column in the source data.
            from_crs: Source CRS (default EPSG:27700, British National Grid).
            to_crs: Target CRS (default EPSG:4326, WGS84).

        Returns:
            DataFrame with new ``lat`` and ``lon`` columns.
        """
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)

        df[easting_col] = pd.to_numeric(
            df[easting_col].astype(str).str.strip(), errors="coerce"
        )
        df[northing_col] = pd.to_numeric(
            df[northing_col].astype(str).str.strip(), errors="coerce"
        )
        df = df.dropna(subset=[easting_col, northing_col]).copy()

        lon, lat = transformer.transform(
            df[easting_col].to_numpy(), df[northing_col].to_numpy()
        )
        df["lat"] = lat.round(4)
        df["lon"] = lon.round(4)

        if drop_source_cols:
            df = df.drop(columns=[easting_col, northing_col])

        return df

    def df_to_geodataframe(
        self,
        df: pd.DataFrame,
        lon_col: str = "lon",
        lat_col: str = "lat",
    ) -> gpd.GeoDataFrame:
        """Promote a DataFrame to a GeoDataFrame using existing lat/lon columns.

        Args:
            df: DataFrame that already has lat/lon columns (see
                :meth:`coordinates_to_lat_lon`).
            lon_col: Name of the longitude column.
            lat_col: Name of the latitude column.

        Returns:
            GeoDataFrame with ``geometry`` column in EPSG:4326.
        """
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )
        return gdf[gdf.geometry.notna()].copy()

    # ── Filters ───────────────────────────────────────────────────────────────

    def filter_by_date(
        self,
        df: pd.DataFrame,
        date: datetime,
        date_col: str = "Record Last Updated (dd/mm/yyyy)",
    ) -> pd.DataFrame:
        """Keep rows where *date_col* >= *date*.

        Args:
            df: DataFrame with parsed datetime columns.
            date: Lower-bound date (inclusive).
            date_col: Column to filter on.

        Returns:
            Filtered DataFrame.
        """
        return df[df[date_col] >= date].copy()

    def filter_by_status(
        self, df: pd.DataFrame, statuses: list[str]
    ) -> pd.DataFrame:
        """Keep rows whose ``Development Status (short)`` is in *statuses*.

        Common status groups are available as module-level constants:
        ``SUCCESSFUL_DEVELOPMENT_TYPES``, ``UNSUCCESSFUL_DEVELOPMENT_TYPES``,
        ``NEUTRAL_DEVELOPMENT_TYPES``, ``CANCELLED_DEVELOPMENT_TYPES``.

        Args:
            df: REPD DataFrame.
            statuses: List of status strings to retain.

        Returns:
            Filtered DataFrame.
        """
        return df[df["Development Status (short)"].isin(statuses)].copy()

    def filter_by_cancelled(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shorthand for :meth:`filter_by_status` with cancelled statuses."""
        return self.filter_by_status(df, list(CANCELLED_DEVELOPMENT_TYPES))

    def filter_by_successful(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shorthand for :meth:`filter_by_status` with successful statuses."""
        return self.filter_by_status(df, SUCCESSFUL_DEVELOPMENT_TYPES)

    def filter_by_unsuccessful(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shorthand for :meth:`filter_by_status` with unsuccessful statuses."""
        return self.filter_by_status(df, UNSUCCESSFUL_DEVELOPMENT_TYPES)

    def filter_by_planning_authority(
        self, df: pd.DataFrame, planning_authority: str
    ) -> pd.DataFrame:
        """Keep rows matching *planning_authority* exactly.

        Args:
            df: REPD DataFrame.
            planning_authority: Planning authority name as it appears in REPD.

        Returns:
            Filtered DataFrame.
        """
        return df[df["Planning Authority"] == planning_authority].copy()

    def filter_by_technology(
        self, df: pd.DataFrame, technology: str
    ) -> pd.DataFrame:
        """Keep rows matching the given technology type.

        Args:
            df: REPD DataFrame.
            technology: Technology type string, e.g. ``"Solar Photovoltaics"``.

        Returns:
            Filtered DataFrame.
        """
        return df[df["Technology Type"] == technology].copy()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def yearly_status_counts(
        self,
        df: pd.DataFrame,
        status_date_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Count projects reaching each status per calendar year.

        For each entry in *status_date_map*, the year is derived from the
        corresponding event date column (e.g. the year a project became
        Operational). This gives a cohort view of the pipeline — how many
        projects crossed each threshold in a given year.

        Args:
            df: REPD DataFrame with parsed datetime columns (run
                :meth:`convert_datetime` first).
            status_date_map: Mapping of ``{display_label: date_column}``.
                Defaults to the four main pipeline stages below.

        Default map::

            {
                "Operational":          "Operational",
                "Denied":               "Planning Permission Refused",
                "Under Construction":   "Under Construction",
                "Awaiting Construction":"Planning Permission Granted",
            }

        Returns:
            DataFrame with year as integer index, one column per status label,
            and integer counts (0 where no projects exist for that year/label).
        """
        if status_date_map is None:
            status_date_map = {
                "Operational": "Operational",
                "Denied": "Planning Permission Refused",
                "Under Construction": "Under Construction",
                "Awaiting Construction": "Planning Permission Granted",
            }

        frames: list[pd.Series] = []
        for label, col in status_date_map.items():
            if col not in df.columns:
                continue
            yearly = (
                df[df[col].notna()]
                .assign(_year=df[col].dt.year)
                .groupby("_year")
                .size()
                .rename(label)
            )
            yearly.index = yearly.index.astype(int)
            frames.append(yearly)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1).fillna(0).astype(int)
        result.index.name = "year"
        return result.sort_index()

    def get_unique(self, df: pd.DataFrame, column: str) -> set[Any]:
        """Return the set of unique non-null values in *column*.

        Args:
            df: REPD DataFrame.
            column: Column name.

        Returns:
            Set of unique values.
        """
        return set(df[column].dropna().unique())

    def create_geojson(self, df: pd.DataFrame) -> dict[str, Any]:
        """Serialise a processed DataFrame to a GeoJSON FeatureCollection.

        Expects ``lat`` and ``lon`` columns to be present. All other columns
        are written as feature properties; non-serialisable types (e.g.
        ``NaT``, ``NaN``) are converted to ``None``.

        Args:
            df: DataFrame with ``lat`` and ``lon`` columns.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        features = []
        prop_cols = [c for c in df.columns if c not in ("lat", "lon", "geometry")]

        for _, row in df.iterrows():
            props: dict[str, Any] = {}
            for col in prop_cols:
                val = row[col]
                if pd.isna(val) if not isinstance(val, (list, dict)) else False:
                    props[col] = None
                elif isinstance(val, pd.Timestamp):
                    props[col] = val.isoformat()
                else:
                    props[col] = val

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["lon"]), float(row["lat"])],
                    },
                    "properties": props,
                }
            )

        return {"type": "FeatureCollection", "features": features}

    def get_search_query(self, row: pd.Series) -> str:
        """Build a plain-text search query string for a single REPD row.

        Useful for feeding into a web search or news API to find coverage of
        a specific project.

        Args:
            row: A single row from a REPD DataFrame.

        Returns:
            Human-readable search query string.
        """
        updated = row.get("Record Last Updated (dd/mm/yyyy)", "")
        return (
            f"{row['Site Name']}, {row['Country']}, {row['Technology Type']}, "
            f"around {updated}"
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def process_pipeline(
        self,
        date: datetime | None = None,
        planning_authority: str | None = None,
        technology: str | None = None,
        status_filter: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Full ETL pipeline: load → parse dates → filter → convert coords → GeoDataFrame.

        Args:
            date: If provided, keep only rows updated on or after this date.
            planning_authority: If provided, keep only rows for this authority.
            technology: If provided, keep only rows for this technology type.
            status_filter: If provided, keep only rows matching these statuses.

        Returns:
            GeoDataFrame ready for analysis or export.
        """
        df = self.load()
        df = self.convert_datetime(df)

        if date is not None:
            df = self.filter_by_date(df, date=date)
            print(f"After date filter ({date:%Y-%m-%d}): {len(df):,} records")

        if planning_authority is not None:
            df = self.filter_by_planning_authority(df, planning_authority)
            print(f"After authority filter: {len(df):,} records")

        if technology is not None:
            df = self.filter_by_technology(df, technology)
            print(f"After technology filter: {len(df):,} records")

        if status_filter is not None:
            df = self.filter_by_status(df, status_filter)
            print(f"After status filter: {len(df):,} records")

        df = self.coordinates_to_lat_lon(df)
        gdf = self.df_to_geodataframe(df)

        return gdf
