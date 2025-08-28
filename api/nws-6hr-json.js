export default async function handler(req, res) {
  const station = req.query.station || "KNYC";
  const url = `https://www.weather.gov/data/obhistory/${station}.html`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      res.status(resp.status).send(`Fetch error: ${resp.statusText}`);
      return;
    }
    const html = await resp.text();

    // Use regex to capture "6 hr" header position and the same column value
    // First, find the "6 hr" column index
    const headerMatch = html.match(/<th>6 hr(?:<br>)?Max(?:<br>\(&deg;F\))?<\/th>/);
    if (!headerMatch) {
      res.status(500).send("Could not find 6 hr Max header");
      return;
    }

    // Then get rows with that column's value
    const rowMatch = html.match(new RegExp(
      `<td>\\d{1,2}</td>\\s*<td>\\d{2}:\\d{2}</td>\\s*` +  // date & time columns
      `(?:[\\s\\S]*?)<td><font color="red">(\\d{1,3})<\\/font><\\/td>`
    ));
    if (!rowMatch) {
      res.status(500).send("Could not extract 6 hr Max value");
      return;
    }

    const maxVal = Number(rowMatch[1]);

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.json({ value: maxVal });
  } catch (err) {
    res.status(500).send("Error parsing HTML: " + err.message);
  }
}
