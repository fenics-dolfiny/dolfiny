import logging

logger = logging.getLogger("dolfiny")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("dolfiny:%(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
