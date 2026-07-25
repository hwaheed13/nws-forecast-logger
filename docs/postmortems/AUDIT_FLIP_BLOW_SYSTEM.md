# Flip Risk & Blow-Past System Audit

**Date**: April 16, 2026  
**Status**: ⚠️ **CRITICAL INCONSISTENCIES FOUND**

---

## Executive Summary

The **blow-past risk severity system has contradictory messaging** between backend and frontend. The color coding and logic are correct, but descriptions are **inaccurate and misleading**.

---

## Backend Analysis (api/flip-risk.js)

### Blow-Past Risk Calculation ✅ CORRECT
```javascript
const blowRisk = blowPoints >= 3 ? "HIGH"
               : blowPoints >= 2 ? "MEDIUM"
               : "NONE";
```

**Severity Thresholds:**
- **NONE**: 0-1 points = Low probability of exceeding all forecasts
- **MEDIUM**: 2-2.9 points = Moderate probability
- **HIGH**: 3+ points = High probability

**Factors Contributing to blowPoints:**
1. High ensemble spread (> 3°F): +1 point
2. Warm obs trajectory (near predicted high): +2 points
3. Warm 925mb temp (> 55°F): +2 points
4. Warm wind direction: +2 points
5. ML much warmer than NWS (> 3°F): +1 point

**Summary for HIGH** (line 382): `"Blow-past risk — actual may exceed all model forecasts"` ✅ **CORRECT & SEVERE**

### Flip Risk Calculation ✅ CORRECT
```javascript
const flipRisk = flipPoints >= 3 ? "HIGH"
               : flipPoints >= 2 ? "MEDIUM"
               : "LOW";
```

Same point accumulation logic. Summary for HIGH (line 380): `"Structural flip risk — bucket may shift before settlement"` ✅ **CORRECT**

---

## Frontend Analysis (public/index.html)

### Card Display ✅ CORRECT
- **Line 4060**: Only shows card if `blow.level !== 'NONE'` ✓
- **Line 4061-4065**: Card layout and structure ✓
- **Line 4089-4092**: Color coding correct:
  - HIGH: Red (#dc2626, #991b1b) with 🔥
  - MEDIUM: Orange (#d97706, #92400e) with ⚠️
  - NONE: Gray (#9ca3af) - not displayed ✓

### ❌ CRITICAL: Blow-Past Descriptions (Line 4064)
```javascript
${blow.level === 'HIGH' ? 'actual may exceed top of range' : 'elevated risk of exceeding range'}
```

**PROBLEM**: 
- HIGH says "top of range" but backend says "ALL model forecasts" (much more severe)
- MEDIUM and NONE both say "elevated risk" (not differentiated)
- Description is WEAKER than severity level

**What it currently shows:**
- HIGH: "actual may exceed top of range" ← TOO MILD
- MEDIUM: "elevated risk of exceeding range" ← VAGUE
- NONE: Not shown ✓

### ❌ Warning Banner Messaging (Lines 4086-4097)
```javascript
const _exceed = _upperBound ? ` — actual may exceed ${_upperBound}` : '';
```

**PROBLEM**:
- Both HIGH and MEDIUM get identical message
- Message is about exceeding ONE bucket (mild)
- Should differentiate: HIGH = all forecasts, MEDIUM = most forecasts

**Current**: Both show "actual may exceed 78°F" (or whatever upper bound)  
**Should be**: HIGH = "actual may exceed all model forecasts", MEDIUM = "elevated risk of exceeding range"

### Banner Title ✅ CORRECT (Line 4096)
```javascript
${_bpIcon} <strong>Blow-past risk ${blow.level}</strong> — model predicts ${_currBktLabel}${_exceed}
```
Shows the level explicitly, which helps.

---

## Flip Risk System ✅ ALL CORRECT

No issues found. The descriptions, colors, and logic are consistent and accurate.

---

## Summary of Issues

| Issue | Location | Severity | Status |
|-------|----------|----------|--------|
| HIGH description contradicts backend | Line 4064 | CRITICAL | ⚠️ NEEDS FIX |
| MEDIUM/NONE not differentiated | Line 4064 | CRITICAL | ⚠️ NEEDS FIX |
| Warning banner same for HIGH & MEDIUM | Lines 4086-4097 | HIGH | ⚠️ NEEDS FIX |
| Colors/icons correct | Lines 4089-4092 | N/A | ✅ OK |
| Flip risk system | flip-risk.js | N/A | ✅ OK |
| Card display logic | Line 4060 | N/A | ✅ OK |

---

## Fixes Required

### Fix 1: Line 4064 - Descriptions
Change from:
```javascript
${blow.level === 'HIGH' ? 'actual may exceed top of range' : 'elevated risk of exceeding range'}
```

To:
```javascript
${blow.level === 'HIGH' ? 'actual may exceed all forecasts' : blow.level === 'MEDIUM' ? 'elevated risk of blow-past' : ''}
```

### Fix 2: Lines 4086-4097 - Warning banner
Differentiate messages by level:
```javascript
const _exceedMsg = blow.level === 'HIGH' 
  ? ' — actual may exceed all model forecasts'
  : blow.level === 'MEDIUM'
  ? ` — actual may exceed ${_upperBound || 'forecast range'}`
  : '';
```

---

## Verification Checklist

- [ ] Line 4064 fixed with level-specific descriptions
- [ ] Lines 4086-4097 warning message differentiated by severity
- [ ] Text matches backend severity definitions
- [ ] Colors/icons match text (visual hierarchy correct)
- [ ] NONE level hidden (card not shown)
- [ ] MEDIUM level shows orange warning
- [ ] HIGH level shows red warning with 🔥

---

## Notes

- The backend logic and calculations are **airtight and correct**
- The flip risk system is **fully correct with no issues**
- The **only problem is frontend descriptions are weak and contradictory**
- Colors and visual hierarchy are **correct**
- Once descriptions are fixed, the entire system will be **accurate and consistent**
