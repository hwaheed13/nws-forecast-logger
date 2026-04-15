#!/usr/bin/env node
import https from 'https';
import fs from 'fs';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);
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
          data: data.length > 0 ? data : null,
          success: res.statusCode === 200
        });
      });
    });
    request.on('error', (err) => {
      resolve({ name, responseTime: Date.now() - startTime, error: err.message, success: false, data: null });
    });
    request.setTimeout(10000, () => {
      request.destroy();
      resolve({ name, responseTime: Date.now() - startTime, error: 'Timeout', success: false, data: null });
    });
  });
}

function extractTemp(text, source) {
  if (!text) {
    console.log(`[${source}] No text to extract from`);
    return null;
  }

  // Try to extract from raw AFOS format (new endpoint)
  // Format: KNYC DS 1600 15/04 901414/ 670430// 90/ 67//...
  // Pattern: KNYC DS [time] [date] [codes]// [HIGH]/ [LOW]//
  let match = text.match(/KNYC\s+DS.*?\/\/\s+(\d+)\//);
  if (match && match[1]) {
    console.log(`[${source}] Extracted temp: ${match[1]}°F (from AFOS format)`);
    return parseFloat(match[1]);
  }

  // Try to extract from HTML (old endpoint)
  const preMatch = text.match(/<pre[^>]*>([\s\S]*?)<\/pre>/);
  if (preMatch && preMatch[1]) {
    console.log(`[${source}] Found <pre> block, extracting from HTML...`);
    match = preMatch[1].match(/KNYC\s+DS.*?\/\/\s+(\d+)\//);
    if (match && match[1]) {
      console.log(`[${source}] Extracted temp: ${match[1]}°F (from HTML)`);
      return parseFloat(match[1]);
    }
  }

  console.log(`[${source}] Failed to extract temperature. Text length: ${text.length}, First 200 chars: ${text.substring(0, 200)}`);
  return null;
}

async function main() {
  console.log('🌡️  Testing DSM endpoints...');
  const utcTimestamp = getUTCTimestamp();
  const oldEndpoint = `https://mesonet.agron.iastate.edu/wx/afos/p.php?pil=DSMNYC&e=${utcTimestamp}`;
  const newEndpoint = `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=DSMNYC`;

  console.log(`Old endpoint: ${oldEndpoint}`);
  console.log(`New endpoint: ${newEndpoint}\n`);

  const [oldResult, newResult] = await Promise.all([
    fetchEndpoint(oldEndpoint, 'old'),
    fetchEndpoint(newEndpoint, 'new'),
  ]);

  console.log(`\nResults:`);
  console.log(`Old: ${oldResult.responseTime}ms (${oldResult.statusCode}) - success: ${oldResult.success}`);
  console.log(`New: ${newResult.responseTime}ms (${newResult.statusCode}) - success: ${newResult.success}`);

  const oldTemp = oldResult.success ? extractTemp(oldResult.data, 'OLD') : null;
  const newTemp = newResult.success ? extractTemp(newResult.data, 'NEW') : null;

  const winner = oldResult.success && newResult.success 
    ? (oldResult.responseTime < newResult.responseTime ? 'old' : 'new')
    : (oldResult.success ? 'old' : (newResult.success ? 'new' : null));

  console.log(`\n🏆 Winner: ${winner}`);
  console.log(`Temperatures - Old: ${oldTemp}°F, New: ${newTemp}°F`);

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
    old_endpoint_raw: oldResult.data ? oldResult.data.substring(0, 1000) : oldResult.error,
    new_endpoint_raw: newResult.data ? newResult.data.substring(0, 1000) : newResult.error,
  };

  const { error: testError } = await supabase.from('dsm_endpoint_tests').insert([testRecord]);
  if (testError) {
    console.error(`Failed to store test results: ${testError.message}`);
  } else {
    console.log('✅ Test results stored in dsm_endpoint_tests');
  }

  // Update nws_daily_summary with winner's data
  if (winner && ((winner === 'old' && oldTemp) || (winner === 'new' && newTemp))) {
    const winnerTemp = winner === 'old' ? oldTemp : newTemp;
    console.log(`\n📝 Updating nws_daily_summary with ${winner} endpoint (${winnerTemp}°F)...`);

    const { error: upsertError } = await supabase
      .from('nws_daily_summary')
      .upsert({
        city: 'nyc',
        date: today,
        dsm_high: winnerTemp,
        dsm_high_time: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }, {
        onConflict: 'city,date',
      });

    if (upsertError) {
      console.error(`❌ Failed to update nws_daily_summary: ${upsertError.message}`);
    } else {
      console.log(`✅ Updated nws_daily_summary for ${today}`);
    }
  } else {
    console.log(`⚠️  No valid temperature to store (winner: ${winner}, temps: old=${oldTemp}, new=${newTemp})`);
  }

  fs.writeFileSync('dsm-test-results.json', JSON.stringify(testRecord, null, 2));
  console.log(`\n📄 Results saved to dsm-test-results.json`);
}

main().catch(err => {
  console.error(`❌ Fatal error: ${err.message}`);
  process.exit(1);
});
