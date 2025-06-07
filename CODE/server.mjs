import express from 'express';
import axios from 'axios';
import querystring from 'query-string';
import { createObjectCsvWriter } from 'csv-writer';
import path from 'path';
import { fileURLToPath } from 'url';

// Define __dirname manually for ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const clientId = '4032480c0c894954a6157f7500407664'; // Your Client ID
const clientSecret = '6631e8b6c59e4e169b46d7dcccb9c10c'; // Your Client Secret
const redirectUri = 'http://localhost:8888/callback';
const scopes = 'user-top-read'; 

const app = express();

// Step 1: Redirect user to authorize
app.get('/login', (req, res) => {
  const authUrl = `https://accounts.spotify.com/authorize?${querystring.stringify({
    response_type: 'code',
    client_id: clientId,
    scope: scopes,
    redirect_uri: redirectUri,
  })}`;
  res.redirect(authUrl);
});

// Step 2: Handle the callback and get access token
app.get('/callback', async (req, res) => {
  const code = req.query.code || null;

  try {
    const response = await axios.post(
      'https://accounts.spotify.com/api/token',
      querystring.stringify({
        code: code,
        redirect_uri: redirectUri,
        grant_type: 'authorization_code',
      }),
      {
        headers: {
          'Authorization': `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      }
    );

    const { access_token } = response.data;
    res.redirect(`/top-tracks?access_token=${access_token}`);
  } catch (error) {
    console.error(error);
    res.send('Error retrieving access token.');
  }
});

// Step 3: Fetch top tracks and enrich with more attributes
app.get('/top-tracks', async (req, res) => {
  const { access_token } = req.query;

  try {
    const timeRanges = ['short_term', 'medium_term', 'long_term']; // Different periods
    const trackData = [];
    const maxTracks = 1000; // Target number of tracks
    const limit = 50; // Max is 50 for each request

    for (const range of timeRanges) {
      let offset = 0;

      while (offset < maxTracks) {
        const response = await axios.get(`https://api.spotify.com/v1/me/top/tracks`, {
          headers: {
            'Authorization': `Bearer ${access_token}`,
          },
          params: {
            limit: limit,
            offset: offset,
            time_range: range,
          },
        });

        const tracks = response.data.items.map(item => ({
          track_id: item.id,
          track_name: item.name,
          artist: item.artists.map(artist => artist.name).join(', '),
          album: item.album.name,
          genre: 'N/A', // Placeholder, will update later
          release_date: item.album.release_date,
          popularity: item.popularity,
        }));

        trackData.push(...tracks);

        // Break if there are no more tracks to fetch
        if (tracks.length < limit) break; 

        offset += limit; // Increase offset for the next batch
      }
    }

    // Enrich with genre data (based on artist)
    for (let track of trackData) {
      try {
        const trackDetails = await axios.get(`https://api.spotify.com/v1/tracks/${track.track_id}`, {
          headers: { 'Authorization': `Bearer ${access_token}` },
        });

        const artistId = trackDetails.data.artists[0].id;
        const genreResponse = await axios.get(`https://api.spotify.com/v1/artists/${artistId}`, {
          headers: { 'Authorization': `Bearer ${access_token}` },
        });

        track.genre = genreResponse.data.genres.join(', ') || 'Unknown';
      } catch (error) {
        console.error('Error fetching genre:', error);
      }
    }

    // Write to CSV
    const csvWriter = createObjectCsvWriter({
      path: path.join(__dirname, 'top_tracks.csv'),
      header: [
        { id: 'track_name', title: 'Track Name' },
        { id: 'artist', title: 'Artist' },
        { id: 'album', title: 'Album' },
        { id: 'genre', title: 'Genre' },
        { id: 'release_date', title: 'Release Date' },
        { id: 'popularity', title: 'Popularity' },
      ],
    });

    await csvWriter.writeRecords(trackData);
    res.redirect('/download');
  } catch (error) {
    console.error(error);
    res.send('Error fetching top tracks.');
  }
});

// Step 4: Endpoint to download the CSV file
app.get('/download', (req, res) => {
  const filePath = path.join(__dirname, 'top_tracks.csv');
  res.download(filePath, 'top_tracks.csv', (err) => {
    if (err) {
      console.error('Error downloading the file:', err);
      res.status(500).send('Error downloading the file.');
    }
  });
});

app.listen(8888, () => console.log('Server running on http://localhost:8888/login'));
