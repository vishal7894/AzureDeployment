import dash
import dash_bootstrap_components as dbc
import dash_auth

# bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

# auth = dash_auth.BasicAuth(
#     app,
#     {'SigmoidDashboard': 'Sigmoid@123'}
# )