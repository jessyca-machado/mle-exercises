"""
Utils para logging, garantindo que model_name esteja sempre presente para o formatter.
"""
from __future__ import annotations
import logging

class EnsureModelNameFilter(logging.Filter):
    """
    Garante que todo LogRecord tenha o campo model_name para não quebrar.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        """Adiciona model_name ao record se não existir, para evitar erros no formatter.
        
        Args:
            record: LogRecord a ser filtrado

        Returns:
            True (sempre permite o log passar)
        """
        if not hasattr(record, "model_name"):
            record.model_name = "-"
        return True


class ModelLogger(logging.LoggerAdapter):
    """
    Inclui model_name em 'extra' para aparecer no formatter.
    """
    def process(self, msg, kwargs):
        """
        Adiciona model_name em extra para o formatter.
        
        Args:            
            msg: mensagem original
            kwargs: kwargs originais do log, pode conter 'extra' com outras chaves

        Returns:
            msg: mensagem inalterada
            kwargs: kwargs com 'extra' atualizado para incluir 'model_name'
        """
        kwargs.setdefault("extra", {})
        kwargs["extra"]["model_name"] = self.extra.get("model_name", "-")

        return msg, kwargs


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Configura o logger com ModelLogger e EnsureModelNameFilter.
    
    Args:
        name: nome do logger (default: __name__)
        level: nível de log (default: logging.INFO)

    Returns:
        Logger configurado para incluir model_name
    """
    logger = logging.getLogger(name)

    # IMPORTANTE: adiciona filtro no root (pega tudo)
    root = logging.getLogger()
    has_filter = any(isinstance(f, EnsureModelNameFilter) for f in root.filters)
    if not has_filter:
        root.addFilter(EnsureModelNameFilter())

    return logger
