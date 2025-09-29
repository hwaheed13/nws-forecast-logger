// api/predict-ml.js
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  
  try {
    const features = req.body;
    
    // Check if models exist
    const modelsExist = 
      fs.existsSync(path.join(process.cwd(), 'temp_model.pkl')) &&
      fs.existsSync(path.join(process.cwd(), 'bucket_model.pkl'));
    
    if (!modelsExist) {
      return res.status(503).json({ 
        error: 'Models not trained yet',
        fallback: true 
      });
    }
    
    // Run Python inference script
    const prediction = await runPythonInference(features);
    
    res.status(200).json(prediction);
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Prediction failed',
      fallback: true,
      details: error.message 
    });
  }
}

async function runPythonInference(features) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python3', [
      path.join(process.cwd(), 'predict.py'),
      JSON.stringify(features)
    ]);
    
    let result = '';
    let error = '';
    
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(error || 'Python process failed'));
      } else {
        try {
          resolve(JSON.parse(result));
        } catch (e) {
          reject(new Error('Invalid JSON from Python: ' + result));
        }
      }
    });
  });
}
