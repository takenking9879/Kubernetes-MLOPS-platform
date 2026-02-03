import yaml
import logging

class BaseUtils:
    """
    Clase base con métodos utilitarios para cargar parámetros y usar logging.
    """
    def __init__(self, logger: logging.Logger, params_path: str):
        self.logger = logger
        self.params_path = params_path

    def load_params(self) -> dict:
        """
        Carga un archivo YAML y retorna un diccionario con los parámetros.
        """
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', self.params_path)
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