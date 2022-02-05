import numpy as np
class Config:
    # API_URL = 'https://hiro.ngrok.io'
    COLOR_API_URL = 'http://128.84.10.197:5005' # COLOR
    NEEDFINDING_API_URL = 'http://128.84.10.197:5006' # NEEDFINDING
    FALLBACK_API_URL = NEEDFINDING_API_URL
    NEW_CARD_ZONE = None
    SEARCH_POS = np.array([[0],[200],[200]])

