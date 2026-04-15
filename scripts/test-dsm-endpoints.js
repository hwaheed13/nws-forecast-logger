#!/usr/bin/env node
import https from 'https';
import fs from 'fs';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
const today = new Date().toISOString().split('T')[0];

function getUTCTimestamp() {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, '0');
  const day = String(now.getUTCDate()).padStart(2, '0');
  const hours = String(now.getUTCHours()).padStart(2, '0');
  const minutes = String(now.getUTCMinutes()).padStart(2, '0');
  return `${year}${month}${day}${hours}${minutes}`;
}

function fetchEndpoint(url, name) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let data = '';
    const request = https.get(url, (res) => {
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        resolve({
          name,
          responseTime: Date.now() - startTime,
          statusCode: res.statusCode,
          data,
          success: res.statusCode === 200
        });
      });
    });
    request.on('error', (err) => {
      resolve({ name, responseTime: Date.now() - startTime, error: err.message, success: false });
    });
    request.setTimeout(10000, () => {
      request.destroy();
      resolve({ name, responseTime: Date.now() - startTime, error: 'Timeout', success: false });
    });
  });
}

function extractTemp(text) {
  if (!text) return null;
  const match = text.match(/KNYC\s+DS\s+\d+\s+\d+\/\d+\s+\d+\s+\d+\/\s+(\d+)\//);
  return match ? parseFloat(match[1]) : null;
}

async function main() {
  console.log('Testing DSM endpoints...');
  const utcTimestamp = getUTCTimestamp();
  const oldEndpoint = `https://mesonet.agron.iastate.edu/wx/afos/p.php?pil=DSMNYC&e=${utcTimestamp}`;
  const newEndpoint = `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=DSMNYC`;

  const [oldResult, newResult] = await Promise.all([
    fetchEndpoint(oldEndpoint, 'old'),
    fetchEndpoint(newEndpoint, 'new'),
  ]);

  console.log(`Old endpoint: ${oldResult.responseTime}ms`);
  console.log(`New endpoint: ${newResult.responseTime}ms`);

  const oldTemp = oldResult.success ? extractTemp(oldResult.data) : null;
  const newTemp = newResult.success ? extractTemp(newResult.data) : null;

  const winner = oldResult.success && newResult.success 
    ? (oldResult.responseTime < newResult.responseTime ? 'old' : 'new')
    : (oldResult.success ? 'old' : 'new');

  console.log(`Winner: ${winner}`);

  const testRecord = {
    test_date: today,
    test_time: new Date().toISOString(),
    old_endpoint_response_ms: oldResult.responseTime,
    new_endpoint_response_ms: newResult.responseTime,
    winner,
    old_endpoint_success: oldResult.success,
    new_endpoint_success: newResult.success,
    old_endpoint_high: oldTemp,
    new_endpoint_high: newTemp,
  };

  const { error } = await supabase.from('dsm_endpoint_tests').insert([testRecord]);
  if (error) {
    console.error(`Failed to store: ${error.message}`);
    process.exit(1);
  }

  console.log('Results stored in Supabase');
  fs.writeFileSync('dsm-test-results.json', JSON.stringify(testRecord, null, 2));
}

main().catch(err => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
