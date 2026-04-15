#!/usr/bin/env node

/**
 * Test DSM endpoints in parallel to compare response times
 * Fetches from both old and new Iowa Mesonet endpoints
 * Stores results in Supabase dsm_endpoint_tests table
 */

const https = require('https');
const fs = require('fs');
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  console.error('❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Get today's date in ISO format
const today = new Date().toISOString().split('T')[0];

// Format timestamp for old endpoint: YYYYmmddHHMI
function getUTCTimestamp() {
  const now = new Date();
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, '0');
  const day = String(now.getUTCDate()).padStart(2, '0');
  const hours = String(now.getUTCHours()).padStart(2, '0');
  const minutes = String(now.getUTCMinutes()).padStart(2, '0');
  return `${year}${month}${day}${hours}${minutes}`;
}

/**
 * Fetch from endpoint with timing
 */
function fetchEndpoint(url, name) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    let data = '';
    let statusCode = null;

    const request = https.get(url, (res) => {
      statusCode = res.statusCode;
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        const responseTime = Date.now() - startTime;
        resolve({
          name,
          url,
          statusCode,
          responseTime,
          data,
          success: statusCode === 200,
        });
      });
    });

    request.on('error', (err) => {
      const responseTime = Date.now() - startTime;
      resolve({
        name,
        url,
        statusCode: null,
        responseTime,
        data: null,
        error: err.message,
        success: false,
      });
    });

    request.setTimeout(10000, () => {
      request.destroy();
      const responseTime = Date.now() - startTime;
      resolve({
        name,
        url,
        statusCode: null,
        responseTime,
        data: null,
        error: 'Timeout',
        success: false,
      });
    });
  });
}

/**
 * Extract DSM high temperature from raw text
 * Format: KNYC DS 1500 14/04 871410/ 640518// 87/ 64//...
 * The high temp is in the format: HH/ (e.g., 87/)
 */
function extractTempFromRawText(text) {
  if (!text) return null;
  // Look for pattern like "KNYC DS ... HH/ ..." where HH is 2 digits
  const match = text.match(/KNYC\s+DS\s+\d+\s+\d+\/\d+\s+\d+\s+\d+\/\s+(\d+)\//);
  if (match && match[1]) {
    return parseFloat(match[1]);
  }
  return null;
}

/**
 * Extract DSM high temperature from HTML
 * Look for text inside <pre> tags containing KNYC data
 */
function extractTempFromHtml(html) {
  if (!html) return null;
  // Extract text from <pre> tags
  const preMatch = html.match(/<pre[^>]*>([\s\S]*?)<\/pre>/);
  if (preMatch && preMatch[1]) {
    return extractTempFromRawText(preMatch[1]);
  }
  return null;
}

/**
 * Main function
 */
async function main() {
  console.log('🌡️  Starting DSM endpoint comparison test...');
  console.log(`📅 Test date: ${today}`);
  console.log(`⏰ Test time: ${new Date().toISOString()}`);

  const utcTimestamp = getUTCTimestamp();

  // Endpoints to test
  const oldEndpoint = `https://mesonet.agron.iastate.edu/wx/afos/p.php?pil=DSMNYC&e=${utcTimestamp}`;
  const newEndpoint = `https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=DSMNYC`;

  console.log(`\n🔄 Fetching from both endpoints in parallel...`);
  console.log(`   Old: ${oldEndpoint}`);
  console.log(`   New: ${newEndpoint}`);

  // Fetch both endpoints simultaneously
  const [oldResult, newResult] = await Promise.all([
    fetchEndpoint(oldEndpoint, 'old'),
    fetchEndpoint(newEndpoint, 'new'),
  ]);

  console.log(`\n✅ Both requests completed`);
  console.log(`   Old endpoint: ${oldResult.responseTime}ms (${oldResult.statusCode})`);
  console.log(`   New endpoint: ${newResult.responseTime}ms (${newResult.statusCode})`);

  // Parse temperatures
  const oldTemp = oldResult.success ? extractTempFromHtml(oldResult.data) : null;
  const newTemp = newResult.success ? extractTempFromRawText(newResult.data) : null;

  console.log(`\n🌡️  Extracted temperatures:`);
  console.log(`   Old endpoint: ${oldTemp}°F`);
  console.log(`   New endpoint: ${newTemp}°F`);

  // Determine winner (faster response)
  let winner = null;
  if (oldResult.success && newResult.success) {
    winner = oldResult.responseTime < newResult.responseTime ? 'old' : 'new';
    console.log(`\n🏆 Winner: ${winner.toUpperCase()} endpoint (${Math.min(oldResult.responseTime, newResult.responseTime)}ms)`);
  } else if (oldResult.success) {
    winner = 'old';
    console.log(`\n⚠️  Only old endpoint succeeded`);
  } else if (newResult.success) {
    winner = 'new';
    console.log(`\n⚠️  Only new endpoint succeeded`);
  } else {
    console.error(`\n❌ Both endpoints failed!`);
    console.error(`   Old error: ${oldResult.error}`);
    console.error(`   New error: ${newResult.error}`);
  }

  // Store results in Supabase
  console.log(`\n📊 Storing results in Supabase...`);

  const testRecord = {
    test_date: today,
    test_time: new Date().toISOString(),
    old_endpoint_response_ms: oldResult.responseTime,
    new_endpoint_response_ms: newResult.responseTime,
    winner: winner,
    old_endpoint_success: oldResult.success,
    new_endpoint_success: newResult.success,
    old_endpoint_high: oldTemp,
    new_endpoint_high: newTemp,
    old_endpoint_raw: oldResult.success ? oldResult.data.substring(0, 500) : oldResult.error,
    new_endpoint_raw: newResult.success ? newResult.data.substring(0, 500) : newResult.error,
  };

  const { data, error } = await supabase
    .from('dsm_endpoint_tests')
    .insert([testRecord]);

  if (error) {
    console.error(`❌ Failed to store results: ${error.message}`);
    process.exit(1);
  }

  console.log(`✅ Results stored in dsm_endpoint_tests table`);

  // If we have a winner, also update the main nws_daily_summary table
  if (winner && ((winner === 'old' && oldTemp) || (winner === 'new' && newTemp))) {
    const winnerTemp = winner === 'old' ? oldTemp : newTemp;
    console.log(`\n📝 Updating nws_daily_summary with ${winner} endpoint data (${winnerTemp}°F)...`);

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
      console.error(`⚠️  Failed to update nws_daily_summary: ${upsertError.message}`);
    } else {
      console.log(`✅ Updated nws_daily_summary table`);
    }
  }

  // Write results to artifact
  fs.writeFileSync('dsm-test-results.json', JSON.stringify(testRecord, null, 2));
  console.log(`\n📄 Results saved to dsm-test-results.json`);

  console.log(`\n✨ Test completed successfully!`);
}

main().catch((err) => {
  console.error(`❌ Fatal error: ${err.message}`);
  process.exit(1);
});
