import plotly.io as pio

def configure_renderers():
    pio.renderers.default = "notebook+plotly_mimetype"
    