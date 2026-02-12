import yaml
import logging
from datetime import datetime, date, timezone
from typing import Optional, Union, Any

class BaseUtils:
    """
    Clase base con métodos utilitarios para cargar parámetros y usar logging.
    """
    def __init__(self, logger: logging.Logger, params_path: str):
        self.logger = logger
        self.params_path = params_path

    @staticmethod
    def format_iceberg_ts(ts: Optional[Union[str, int, float, datetime, date]]) -> Optional[str]:
        """
        Normalize a timestamp to an ISO-8601 UTC string compatible with
        Iceberg's `timestamp with time zone` (timestamptz).

        Supported inputs:
        - ISO strings: "YYYY-MM-DDTHH:MM:SS", with or without offset (e.g. "+01:00")
        - Space-separated: "YYYY-MM-DD HH:MM:SS"
        - Strings ending with "Z" (UTC)
        - `datetime` (aware or naive)
        - `date` (treated as midnight UTC)
        - Unix epoch seconds (int or float)

        Returns:
            ISO-8601 string with UTC offset, e.g. "2026-01-01T00:30:00+00:00", or `None`.
        Raises:
            ValueError on unparseable inputs.
        """
        if ts is None:
            return None

        try:
            # Fast-path: datetime objects
            if isinstance(ts, datetime):
                dt = ts
            elif isinstance(ts, date):
                # date -> midnight
                dt = datetime(ts.year, ts.month, ts.day)
            elif isinstance(ts, (int, float)):
                # epoch seconds -> aware UTC datetime
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            else:
                s = str(ts).strip()
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                if " " in s and "T" not in s:
                    s = s.replace(" ", "T", 1)
                dt = datetime.fromisoformat(s)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt.astimezone(timezone.utc).isoformat()

        except Exception as exc:
            raise ValueError(f"Could not parse timestamp {ts!r}: {exc}") from exc

    def load_params(self) -> dict:
        """
        Carga un archivo YAML y retorna un diccionario con los parámetros sanitizados.
        """
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            
            # Auto-sanitize splits timestamps to Iceberg ISO-8601 format
            for split in params.get('splits', {}).values():
                if isinstance(split, dict):
                    if 'start' in split: split['start'] = self.format_iceberg_ts(split['start'])
                    if 'end' in split: split['end'] = self.format_iceberg_ts(split['end'])

            self.logger.debug('Parameters retrieved and sanitized from %s', self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise