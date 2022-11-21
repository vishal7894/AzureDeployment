from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from pages import Simulation, ErrorMetrics, home

# # building the navigation bar
# dropdown = dbc.DropdownMenu(
#     children=[
#         dbc.DropdownMenuItem("Simulation", href="/Simulation"),
#         dbc.DropdownMenuItem("Prediction Accuracy", href="/ErrorMetrics")
#     ],
#     nav = True,
#     in_navbar = True,
#     label = "Explore",
#     style = {'marginRight': "100px"}
# )

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/PEPSICO_logo.png", height="30px"), width = 2),
                        dbc.Col(dbc.NavbarBrand("PepsiCo Colombia - SOM Prediction Dashboard v1.1", 
                                                style = {'color': '#01529C', 'font-family': 'Verdana',
                                                         'font-weight': 'bold', 
                                                        'marginLeft': '140px',
                                                        'font-size': '180%', 'align': 'center'}), width=10),
                    ],
                    align="center", justify = "center",
                    no_gutters=True
                ),
                # href="/Simulation",
                style={"textDecoration": "none"}
            ),
            # dbc.NavbarToggler(id="navbar-toggler2"),
            # dbc.Collapse(
            #     dbc.Nav(
            #         # right align dropdown menu with ml-auto className
            #         [dropdown], className="ml-auto", navbar=True
            #     ),
            #     id="navbar-collapse2",
            #     navbar=True,
            # ),
        ], fluid= True
    ),
    color="#E4EDED",
    # dark=True,
    # className="mb-4",
)

# def toggle_navbar_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open

# for i in [2]:
#     app.callback(
#         Output(f"navbar-collapse{i}", "is_open"),
#         [Input(f"navbar-toggler{i}", "n_clicks")],
#         [State(f"navbar-collapse{i}", "is_open")],
#     )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/ErrorMetrics':
        return ErrorMetrics.layout
    elif pathname == '/home':
        return home.layout
    else:
        return Simulation.layout

if __name__ == '__main__':
    app.run_server(port=80, host= '0.0.0.0')