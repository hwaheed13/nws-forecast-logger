export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET');
  
  const ACCUWEATHER_KEY = process.env.ACCUWEATHER_API_KEY;
  const LOCATION = (req.query && req.query.location) || '2627448';
  
  if (!ACCUWEATHER_KEY) {
    return res.status(500).json({ error: 'ACCUWEATHER_API_KEY not configured' });
  }

  try {
    const response = await fetch(
      `https://dataservice.accuweather.com/forecasts/v1/daily/5day/${LOCATION}?apikey=${ACCUWEATHER_KEY}`
    );
    
    if (!response.ok) {
      throw new Error(`AccuWeather API returned ${response.status}`);
    }
    
    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    console.error('AccuWeather API error:', error);
    res.status(500).json({ error: 'Failed to fetch AccuWeather data' });
  }
}
