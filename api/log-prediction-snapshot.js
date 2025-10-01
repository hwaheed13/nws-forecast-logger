export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({error: 'Method not allowed'});
  }

  const {target_date, prediction_type, prediction_value, snapshot_time} = req.body;
  
  if (!target_date || !prediction_type || !prediction_value) {
    return res.status(400).json({error: 'Missing required fields'});
  }

  try {
    const {Octokit} = await import('@octokit/rest');
    const octokit = new Octokit({auth: process.env.GITHUB_TOKEN});
    
    const owner = 'hwaheed13';
    const repo = 'nws-forecast-logger';
    const path = 'predictions_snapshot.csv';
    
    // Get current file
    const {data: fileData} = await octokit.repos.getContent({owner, repo, path});
    const content = Buffer.from(fileData.content, 'base64').toString('utf-8');
    
    // Append new row
    const newRow = `\n${snapshot_time},${target_date},${prediction_type},${prediction_value}`;
    const updatedContent = content + newRow;
    
    // Commit
    await octokit.repos.createOrUpdateFileContents({
      owner,
      repo,
      path,
      message: `Log ${prediction_type} prediction for ${target_date}`,
      content: Buffer.from(updatedContent).toString('base64'),
      sha: fileData.sha
    });
    
    res.status(200).json({success: true});
  } catch (error) {
    console.error('GitHub commit failed:', error);
    res.status(500).json({error: error.message});
  }
}
