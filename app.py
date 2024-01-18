# Import packages
#from dash import 
from dash_extensions.enrich import Dash, html, dash_table, dcc, callback, Output, Input, State, DashProxy, Serverside, ServersideOutputTransform
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
import data_classes
import plot_module
import base64
import settings
import json
import copy

PLOTLY_BG_COLOR = '#111111'

# Incorporate data
fig_creator = plot_module.PxFigureCreator()

audio_file = data_classes.AudioFile.from_pathstring(
    '/Users/janprzybyszewski/Documents/AudioViz/Audio/ViolinDuet.wav'
)

# Initialize the app 
app = DashProxy(
    __name__,
    title='AudioViz',
    transforms=[
       # TriggerTransform(),  # enable use of Trigger objects
        ServersideOutputTransform(),  # enable use of ServersideOutput objects
    ]
)

# Page Elements
title_box = html.Div(
    children=[
        dmc.Title('AudioViz', color="white", size="3em", id='main_title'),
    ],
    style={
        'padding': '15px 30px'
    }
)

upload_box = html.Div(
    children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'solid',
                'borderRadius': '25px',
                'borderColor': '#FFFFFF',
                'backgroundColor': PLOTLY_BG_COLOR,
                'textAlign': 'center',
                'margin': '10px',
                'color':'white',
                'font-size':'12px',
                'font-family': 'Arial, Helvetica, sans-serif'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        dcc.Loading(dcc.Store(id="serverside-cache"), fullscreen=True, type="dot"),
    ],
    style={
        'padding': '5px 30px',
    "backgroundColor": "black",
    }
)

file_metadata_box = html.Div(
    children = [
        dmc.Title('File Info', color='white', size='1.25em', id='file_metadata_title')
    ],
    style={
        'padding': '10px 10px 10px',
        #'backgroundColor': 'red',
        'width': '95%',
        'height': '15%',
        }
)

selection_preview_button = html.Div(
    children = [
        html.Button(
            'Preview',
            id='Preview',
            n_clicks=0,
        ),
    ],
    style={
        'textAlign':"center",
        'position': 'absolute',
        'bottom': '20px',
        #'right': '5px',
         'width': '100%',
    }
)


selection_metadata_box = html.Div(
    children = [
        dmc.Title('Selection Info', color='white', size='1.25em', id='selection_metadata_title'),
        html.P(
            className="graph-type", id="graph-type", children=[""],
            style={
                'display':' none',
                'font-size':'0px',
            }
        ),
        #html.P(className='selection-metadata', id='selection-metadata', children=[''],),
        dmc.Text(id='selection-metadata', color='white'),
        selection_preview_button
    ],
    style={
        'padding': '10px 10px 10px',
        #'backgroundColor': 'yellow',
        'width': '90%',
        'height': '70%',
        'position':'relative',
        }
)

metadata_display_box = html.Div(
    children = [
        file_metadata_box,
        selection_metadata_box
    ],
    style={
        'padding': '10px 10px 10px',
        #'backgroundColor': 'blue',
        'width': '95%',
        'height': '100%',
        }
)


plot_selection_bar =  html.Div(
    children=[
        dmc.SegmentedControl(
            id="view-select",
            value="timeDomain",
            data=[
                {"value": "timeDomain", "label": "Time Domain"},
                {"value": "reassigned", "label": "Reassigned Spectrogram"},
                {"value": "clustered", "label": "Data Clusters"},
                {"value": "curves", "label": "Fitted Curves"},
            ],
            mt=10,
        ),   
],
    style={
        'padding': '0px 10px 10px',
    }
)


param_adjustment_div = html.Div(
    children = [
        dmc.Text(
            "Horizontal Grouping Bias",
            weight=700,
            color='white'
        ),
        dmc.Slider(
            id="slider-callback",
            value=26,
            marks=[
                {"value": 5, "label": "Low"},
                {"value": 50, "label": "Medium"},
                {"value": 95, "label": "High"},
            ],
            mb=35,
        ),
    ]
)

graph_container = html.Div(
    children=[
        dcc.Graph(figure={}, id='graph-placeholder'),
        param_adjustment_div
    ],
    style={
        'padding': '15px 30px',
        "backgroundColor": PLOTLY_BG_COLOR,
        #"backgroundColor": 'pink',
        'width': '80em',
        'borderRadius': '25px',
        'textAlign': 'center'
    }
)

footer = dmc.Footer(
    height=20,
    fixed=True,
    children=[
        html.Div([
            dmc.Text("UCD 2024", color='dimmed')
        ])
    ],
    style={"backgroundColor": PLOTLY_BG_COLOR},
)

# App layout
app.layout = dmc.MantineProvider(
    theme={"colorScheme" : "dark",
           "backgroundColor": "black",
           },
    children=[
        html.Div(
            style={"backgroundColor": "black",
                   'height':'100vh',
                   #'margin':'-10px',
                   #'padding':'-10px'
                   },
            children=[
                # --- Container Start ---

                title_box,
                    
                dmc.Grid(
                    children=[
                        # Left Column
                        dmc.Col([
                            upload_box,

                            metadata_display_box
                        ], span=2),

                        # Right Column
                        dmc.Col([
                            plot_selection_bar,
                            graph_container,
                        ], span=8),
                        # End of Grid
                    ],
                    style={
                        'padding': '0px 0px',
                       # "backgroundColor": "yellow",
                    }
                ),

                footer
                # --- Container End ---
            ],
        )            
    ],
)

def create_figure_cache(audio_file: data_classes.AudioFile,
                        prescale_x: int = 120):
    fig_creator = plot_module.PxFigureCreator()

    points = audio_file.get_reassigned_points()
    
    group_factory = data_classes.ClusterGroupFactory(points, prescale_x)

    cluster_group = group_factory.get_cluster_group()
    cluster_group_copy = copy.deepcopy(cluster_group)
    stack_groups = cluster_group.get_harmonic_stack_groups()

    figure_cache = {
        "time_domain_plot":fig_creator.plot_time_domain(audio_file),
        "reassigned_plot":fig_creator.plot_reassigned_scatter(audio_file),
        "clustered_plot":fig_creator.plot_clusters(audio_file),
        "curves_plot":fig_creator.plot_curves(audio_file),
        "cluster_group":cluster_group_copy,
        "stack_groups":stack_groups,
    }

    return figure_cache

# Callback for file upload
@callback(
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    Output("serverside-cache", "data"),
)
def upload_and_store(input_upload_data, upload_filename):

    if input_upload_data is not None:
        upload_data_type, upload_data = input_upload_data.split(',')
    else:
        upload_data_type = None
        upload_data = None
        
    print(f"Upload and store Called with data of type ({upload_data_type}) and filename ({upload_filename})!")

    if upload_data is None:
        # Incorporate data
        fig_creator = plot_module.PxFigureCreator()

        audio_file = data_classes.AudioFile.from_pathstring(
            '/Users/janprzybyszewski/Documents/AudioViz/Audio/ViolinDuet.wav'
        )

        figure_cache = create_figure_cache(audio_file)
        
        output = Serverside(figure_cache)
        return output

    # Decode the input data from base64
    upload_bytes = base64.b64decode(upload_data + '==')

    # Turn the input data into the an AudioFile Object
    audio_file = data_classes.AudioFile.from_bytes(upload_bytes)

    # Create Dict of Plots
    fig_creator = plot_module.PxFigureCreator()
    figure_cache = create_figure_cache(audio_file)
    output = Serverside(figure_cache)
    return output

# Callback for plot mode selections
@callback(
    Output(component_id='graph-placeholder', component_property='figure'),
    Output(component_id='graph-type', component_property='children'),
    Input(component_id='view-select', component_property='value'),
    State("serverside-cache", "data")
)
def update_graph(fig_type, stored_data):
    print(f"Update Graph call with type {fig_type}")

    if stored_data is not None:
        match fig_type:
            case "timeDomain":
                fig = stored_data['time_domain_plot']
            case "reassigned":
                fig = stored_data['reassigned_plot']
            case "clustered":
                fig = stored_data['clustered_plot']
            case "curves":
                fig = stored_data['curves_plot']
            case _:
                # This should never happen
                pass
    else:
        fig = fig_creator.plot_time_domain(audio_file)
        return fig, 'reassigned'
        

    print(fig_type)
    return fig, fig_type

# Callback for slider param adjustment
def update_horizontal_grouping():
    pass

# Callback for graph click interaction
@callback(
    Output('selection-metadata', 'children'),
    Input('graph-placeholder', 'clickData'),
    Input(component_id='graph-type', component_property='children'),
    State("serverside-cache", "data")
)
def display_click_data(clickData, mode, stored_data):
    print(f'Display click data call with data {clickData} and mode {mode}')
    if clickData is None:
        return "None Selected"

    if stored_data is None:
        return "None Selected"
    
    click_data_extracted = clickData['points'][0]
    x_value = click_data_extracted['x']
    y_value = click_data_extracted['y']
    curve_number = click_data_extracted['curveNumber']
    
    time_location_string = f'    {x_value} s.\n'
    frequency_location_string = f'    {y_value} Hz\n'
    location_string = "Point located at: " \
        + time_location_string \
        + frequency_location_string
    
    # clickData parsing should vary with
    # the mode of the graph
    show_location_string = False
    show_cluster_data = False

    match mode:
        case "timeDomain":
            pass
        case "reassigned":
            show_location_string = True
        case "clustered":
            show_location_string = True
            show_cluster_data = True
        case _: # use default case for curve display
            show_cluster_data = True

    # Init output string
    outstring = location_string

    # Extract cluster in question
    if show_cluster_data:
        cluster_group = stored_data["cluster_group"]
        cluster_info = cluster_group.query_member(curve_number)

        cluster_string = '\n' + str(cluster_info)
        outstring += cluster_string
        
    print(clickData['points'][0])

    return outstring



# Run the App
if __name__ == '__main__':
    app.run(debug=True)
