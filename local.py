# -*- coding: utf-8 -*-


import base64
import os
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

from PIL import Image
from io import BytesIO



IMAGE_DATASETS = ('mnist_3000', 'modu_project')

with open('demo_description.md', 'r') as file:
    demo_md = file.read()


def merge(a, b):
    return dict(a, **b)

def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}

def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64

def Card(children, **kwargs):
    return html.Section(
        children,
        style=merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **omit(['style'], kwargs)
    )

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={'margin': '25px 5px 30px 0px'},
        children=[
            f"{name}:",
            html.Div(style={'margin-left': '5px'}, children=[
                dcc.Slider(id=f'slider-{short}',
                           min=min,
                           max=max,
                           marks=marks,
                           step=step,
                           value=val)
            ])
        ])





local_layout = html.Div(
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem',
        'padding': '10px 30px'
    },
    children=[
        # Header
        html.Div(className="row", children=[
            html.H2(
                'Sound Lab - Team C',
                id='title',
                style={
                    'float': 'left',
                    'margin-top': '20px',
                    'margin-bottom': '0',
                    'margin-left': 70
                }
            ),

            html.A([
                html.Img(
                    src='https://is3-ssl.mzstatic.com/image/thumb/Purple113/v4/88/a0/13/88a013f0-2c7e-e421-fe42-a775f42d2bbb/source/512x512bb.jpg',
                    style={
                        'width': '5%',  # 너비 기준으로 조절하는 것을 추천
                        'float': 'right',  # 위치
                        'position': 'relative',  # 다른 요소의 위치 따라 상대적으로 배치
                        'padding-top': 1,  # 윗 경계면으로부터 간격
                        'padding-right': 45,  # 우측 경계면으로부터 간격
                    },
                )
            ], href='https://www.notion.so/Documentation-6fd5abe0e947489a9be98ede3678fb68'
            ),
        ]),

        # Body
        html.Div(className="row", children=[
            html.Div(className="eight columns", children=[
                dcc.Graph(
                    id='graph-3d-plot-tsne',
                    style={'height': '98vh'}
                )
            ]),

            html.Div(className="four columns", children=[
                Card([

                    dcc.Markdown('''
**TRY DIFFERENT DATASETS & PARAMETERS**
'''),

                    dcc.Dropdown(
                        id='dropdown-dataset',
                        searchable=False,
                        options=[
                            # TODO: Generate more data
                            {'label': 'MNIST Digits', 'value': 'mnist_3000'},
                            {'label': 'Sound Lab - Team C', 'value': 'modu_project'},
                        ],
                        value='mnist_3000',
                        placeholder="Select a dataset"
                    ),

                    NamedSlider(
                        name="Number of Iterations",
                        short="iterations",
                        min=250,
                        max=1000,
                        step=None,
                        val=500,
                        marks={i: i for i in [ 250, 500, 1000]}
                    ),

                    NamedSlider(
                        name="Perplexity",
                        short="perplexity",
                        min=1,
                        max=100,
                        step=None,
                        val=30,
                        marks={i: i for i in [1, 3, 5, 10, 30, 50, 100]}
                    ),

                    NamedSlider(
                        name="Initial PCA Dimensions",
                        short="pca-dimension",
                        min=30,
                        max=100,
                        step=None,
                        val=50,
                        marks={i: i for i in [30, 50, 100]}
                    ),

                    NamedSlider(
                        name="Learning Rate",
                        short="learning-rate",
                        min=10,
                        max=200,
                        step=None,
                        val=100,
                        marks={i: i for i in [10, 50, 100, 200]}
                    ),

                ]),

                Card(style={'padding': '5px'}, children=[
                    html.Div(id='div-plot-click-message',
                             style={'text-align': 'center',
                                    'margin-bottom': '7px',
                                    'font-weight': 'bold'}
                             ),

                    html.Div(id='div-plot-click-image'),

                    html.Div(id='div-plot-click-wordemb')
                ])
            ])
        ]),

        dcc.Markdown('''
* * * *
        '''),

        # Demo Description
        html.Div(
            className='row',
            children=html.Div(
                style={
                    'width': '75%',
                    'margin': '30px auto',
                },
                children=dcc.Markdown(demo_md)
            )
        ),

        html.Div([
            dcc.Markdown('''
**This is the modified version of the t-SNE explorer. To view the source code, please visit the app's [GitHub Repository](https://github.com/plotly/dash-tsne) of the original version.
To view pre-generated simulations of t-SNE on MNIST, Fashion MNIST, CIFAR10, and Word Embeddings check out the [demo app](https://dash-tsne.herokuapp.com).**
''')],
            style={
                'width': '75%',
                'margin': '30px auto',
            },
            className="row"
        ),

    ]
)



def local_callbacks(app):

    # Demo ver.
    def generate_figure_image(groups, layout):
        data = []

        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val['x'],
                y=val['y'],
                z=val['z'],
                text=[idx for _ in range(val['x'].shape[0])],
                textposition='top',
                mode='markers',
                marker=dict(
                    size=4.5,
                    symbol='circle'
                )
            )
            data.append(scatter)

        figure = go.Figure(
            data=data,
            layout=layout
        )

        return figure


    @app.server.before_first_request
    def load_image_data():
        global data_dict

        data_dict = {
            'mnist_3000': pd.read_csv("data/mnist_3000_input.csv"),
            'modu_project': pd.read_csv("data/modu_project_fileName_list.csv"),
        }


    @app.callback(Output('graph-3d-plot-tsne', 'figure'),
                  [Input('dropdown-dataset', 'value'),
                   Input('slider-iterations', 'value'),
                   Input('slider-perplexity', 'value'),
                   Input('slider-pca-dimension', 'value'),
                   Input('slider-learning-rate', 'value'),
                   ]
                  )


    def display_3d_scatter_plot(dataset, iterations, perplexity, pca_dim, learning_rate):
        if dataset:
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv', index_col=0, encoding="ISO-8859-1")

            except FileNotFoundError as error:
                print(error, "\nThe dataset currently not available")
                figure = go.Figure()
                return figure

            # Plot layout
            axes = dict(
                title='',
                showgrid=True,
                zeroline=False,
                showticklabels=False
            )

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=axes,
                    yaxis=axes,
                    zaxis=axes
                ),
                legend = dict(x=.9, y=0.5)
            )

            # For Image datasets
            if dataset in IMAGE_DATASETS:

                if dataset == 'mnist_3000':
                    embedding_df['label'] = embedding_df.index

                groups = embedding_df.groupby('label')
                figure = generate_figure_image(groups, layout)

            else:
                # figure = go.Figure(data=[empty_trace], layout=empty_layout)
                figure = go.Figure()

            return figure

    @app.callback(Output('div-plot-click-image', 'children'),
                  [Input('graph-3d-plot-tsne', 'clickData'),
                   Input('dropdown-dataset', 'value'),
                   Input('slider-iterations', 'value'),
                   Input('slider-perplexity', 'value'),
                   Input('slider-pca-dimension', 'value'),
                   Input('slider-learning-rate', 'value')])
    def display_click_image(clickData,
                            dataset,
                            iterations,
                            perplexity,
                            pca_dim,
                            learning_rate):
        if dataset in IMAGE_DATASETS and clickData:
            # Load the same dataset as the one displayed
            path = f'demo_embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}'

            try:
                embedding_df = pd.read_csv(path + f'/data.csv', encoding="ISO-8859-1")

            except FileNotFoundError as error:
                print(error, "\nThe dataset currently not available.")
                return

            # Convert the point clicked into float64 numpy array
            click_point_np = np.array([clickData['points'][0][i] for i in ['x', 'y', 'z']]).astype(np.float64)

            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = embedding_df.loc[:, 'x':'z'].eq(click_point_np).all(axis=1)

            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():

                clicked_idx = embedding_df[bool_mask_click].index[0]

                if dataset == 'mnist_3000':
                    # Retrieve the image corresponding to the index
                    image_vector = data_dict[dataset].iloc[clicked_idx]
                    image_np = image_vector.values.reshape(28, 28).astype(np.float64)

                    # Encode image into base 64
                    image_b64 = numpy_to_b64(image_np)

                    return html.Img(
                        src='data:image/png;base64, ' + image_b64,
                        style={
                            'height': '25vh',
                            'display': 'block',
                            'margin': 'auto'
                        }
                    )

                elif dataset == 'modu_project':
                    file_name = data_dict[dataset].loc[clicked_idx, 'fileName'].replace('.jpg', '.png')
                    test_base64 = base64.b64encode(open(os.path.join('data/modu_project_images/', file_name), 'rb').read()).decode('ascii')

                    return html.Img(
                        src='data:image/png;base64,{}'.format(test_base64),
                        style={
                            'height': '25vh',
                            'display': 'block',
                            'margin': 'auto'
                        }
                    ),



        return None

    @app.callback(Output('div-plot-click-message', 'children'),
                  [Input('graph-3d-plot-tsne', 'clickData'),
                   Input('dropdown-dataset', 'value')])
    def display_click_message(clickData, dataset):
        """
        Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        :param clickData:
        :param dataset:
        :return:
        """
        if dataset in IMAGE_DATASETS:
            if clickData:
                return "Image Selected"
            else:
                return "Click a data point on the scatter plot to display its corresponding image."

