"""repd — Python tools for the UK Renewable Energy Planning Database (REPD).

Quick start::

    from repd.processor import REPDProcessor

    processor = REPDProcessor("path/to/REPD_Publication_Q4_2025.csv")
    gdf = processor.process_pipeline(date=datetime(2025, 1, 1))
"""

from repd.processor import REPDProcessor

__all__ = ["REPDProcessor"]
__version__ = "0.1.0"
