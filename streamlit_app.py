import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import base64
from requests import post, get
import json
import csv
from sklearn import preprocessing

client_id = 'f177c75faa7d4fba999b81b922c1a042'
client_secret = '46d8af627ab54f5a9462a50a6505ee6e'
playlistId = '0IN7IWKmIfwlEysGyWUuRg'

dataset = []
dataset2 = []
dataset3 = []

def getToken():
    auth_string = client_id + ':' + client_secret

    auth_b64 = base64.b64encode(auth_string.encode('utf-8'))

    url = 'https://accounts.spotify.com/api/token'

    headers = {
        'Authorization': 'Basic ' + auth_b64.decode('utf-8'),
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {'grant_type': 'client_credentials'}

    result = post(url, headers=headers, data=data)

    json_result = json.loads(result.content)
    token = json_result['access_token']

    return token

def getAuthHeader(token):
    return {'Authorization': 'Bearer ' + token}

def getAudioFeatures(token, trackId):

    url = f'https://api.spotify.com/v1/audio-features/{trackId}'

    headers = getAuthHeader(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)

    audio_features_temp = [
        json_result['danceability'],
        json_result['energy'],
        json_result['key'],
        json_result['loudness'],
        json_result['mode'],
        json_result['speechiness'],
        json_result['acousticness'],
        json_result['instrumentalness'],
        json_result['liveness'],
        json_result['valence'],
        json_result['tempo'],
    ]
    dataset2.append(audio_features_temp)

def getPlaylistItems(token, playlistId):

    url = f'https://api.spotify.com/v1/playlists/{playlistId}/tracks'
    limit = '&limit=100'
    market = '?market=ID'

    fields = '&fields=items%28track%28id%2Cname%2Cartists%2Cpopularity%2C+duration_ms%2C+album%28release_date%29%29%29'
    url = url+market+fields+limit

    headers = getAuthHeader(token)
    result = get(url, headers=headers)
    json_result = json.loads(result.content)

    for i in range(len(json_result['items'])):
        playlist_items_temp = []
        playlist_items_temp.append(json_result['items'][i]['track']['id'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['artists'][0]['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['popularity'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['duration_ms'])
        playlist_items_temp.append(
            int(json_result['items'][i]['track']['album']['release_date'][0:4]))
        dataset.append(playlist_items_temp)

    for i in range(len(json_result['items'])):
        playlist_items_temp = []
        playlist_items_temp.append(json_result['items'][i]['track']['id'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['artists'][0]['name'].encode('utf-8'))
        playlist_items_temp.append(
            json_result['items'][i]['track']['popularity'])
        playlist_items_temp.append(
            json_result['items'][i]['track']['duration_ms'])
        playlist_items_temp.append(
            int(json_result['items'][i]['track']['album']['release_date'][0:4]))
        dataset.append(playlist_items_temp)

    for i in range(len(dataset)):
        getAudioFeatures(token, dataset[i][0])

    for i in range(len(dataset)):
        dataset3.append(dataset[i]+dataset2[i])

    print(dataset3)
    
    with open('dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "name", "artist", "popularity", "duration_ms", "year", "danceability", "energy", "key", "loudness", "mode",
                         "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"])
        writer.writerows(dataset3)

token = getToken() 
print('access token : '+token) 
getPlaylistItems(token, playlistId)

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px 

data = pd.read_csv('dataset.csv')
# data.head()

data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
data['name'] = data['name'].map(lambda x: str(x)[2:-1])

# data.head()

data = data[data['name'] != '']

data = data.reset_index(drop=True)
# data.head()

data2 = data.copy()
data2 = data2.drop(['artist', 'name', 'year', 'popularity', 'key','duration_ms', 'mode', 'id'], axis=1)

# data2.head()

from sklearn import preprocessing

x = data2.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data2 = pd.DataFrame(x_scaled)

data2.columns = ['acousticness','danceability','energy','instrumentalness','loudness', 'liveness', 'speechiness', 'tempo','valence']

# data2.describe()

pca = PCA(n_components=2)
pca.fit(data2)
pca_data = pca.transform(data2)

pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])
# pca_df.head()

fig = px.scatter(pca_df, x='x', y='y', title='PCA')
# fig.show()

data2 = list(zip(pca_df['x'], pca_df['y']))

kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)

fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_, color_continuous_scale='rainbow', hover_data=[data.artist, data.name])
# fig.show()

def dataProcessing():
    data = pd.read_csv('dataset.csv')
    data
    st.write("## Preprocessing Result")

    data = data[['artist', 'name', 'year', 'popularity', 'key', 'mode', 'duration_ms', 'acousticness',
                'danceability', 'energy', 'instrumentalness', 'loudness', 'liveness', 'speechiness', 'tempo', 'valence']]
    data = data.drop(['mode'], axis=1)
    data['artist'] = data['artist'].map(lambda x: str(x)[2:-1])
    data['name'] = data['name'].map(lambda x: str(x)[2:-1])
    st.write("### Data to be deleted:")
    data[data['name'] == '']
    data = data[data['name'] != '']

    st.write("## Normalization Result")
    data2 = data.copy()
    data2 = data2.drop(
        ['artist', 'name', 'year', 'popularity', 'key', 'duration_ms'], axis=1)
    x = data2.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data2 = pd.DataFrame(x_scaled)
    data2.columns = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'loudness', 'liveness', 'speechiness', 'tempo', 'valence']
    data2

    st.write("## Dimensionality Reduction with PCA")
    pca = PCA(n_components=2)
    pca.fit(data2)
    pca_data = pca.transform(data2)
    pca_df = pd.DataFrame(data=pca_data, columns=['x', 'y'])
    fig = px.scatter(pca_df, x='x', y='y', title='PCA')
    st.plotly_chart(fig)

    st.write("## Clustering with K-Means")
    data2 = list(zip(pca_df['x'], pca_df['y']))
    kmeans = KMeans(n_init=10, max_iter=1000).fit(data2)
    fig = px.scatter(pca_df, x='x', y='y', color=kmeans.labels_,
                     color_continuous_scale='rainbow', hover_data=[data.artist, data.name])
    st.plotly_chart(fig)

    st.write("Enjoy!")

st.write("# Spotify Playlist Clustering")
client_id = st.text_input("Enter Client ID")
client_secret = st.text_input("Enter Client Secret")
playlistId = st.text_input("Enter Playlist ID")

if st.button('Create Dataset!'):
    token = getToken()
    getPlaylistItems(token, playlistId)