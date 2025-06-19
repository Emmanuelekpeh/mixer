// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Mixing workflow tests - testing the complete user flow
 */
test.describe('Mixing Workflow', () => {test('should complete full mixing session creation flow', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Step 1: Artist Name Entry  
    console.log('🎤 Testing artist name entry...');
    
    const nameInput = page.locator('input[placeholder*="Enter your name"]');
    await expect(nameInput).toBeVisible();
    await nameInput.fill('TestArtist_' + Date.now());

    // Step 2: File Upload Test
    console.log('📁 Testing file upload...');
    
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeVisible();
    
    // Create a test audio file (empty MP3-like file for testing)
    const testFile = Buffer.from('test audio data');
    await fileInput.setInputFiles([{
      name: 'test-audio.mp3',
      mimeType: 'audio/mpeg',
      buffer: testFile
    }]);
    
    // Wait for file to be processed
    await page.waitForTimeout(2000);

    // Step 3: Start Mixing Session
    console.log('🚀 Testing mixing session start...');
    
    const startBtn = page.locator('button:has-text("START MIXING SESSION")');
    await expect(startBtn).toBeVisible();
    await startBtn.click();
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);

    // Step 4: Verify Session Started
    console.log('✅ Verifying mixing session started...');
    
    // Should navigate to a different page or show different content
    // Check if we're still on the same page or moved to a battle/results page
    const currentUrl = page.url();
    console.log('Current URL after start:', currentUrl);
      // Should not have error messages
    await expect(page.locator('.error, [class*="error"]')).not.toBeVisible();
    await expect(page.locator('text=Error')).not.toBeVisible();
  });

  test('should handle audio playback in battles', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
      // Test navigation to battle arena (if available)
    const battleLink = page.locator('a[href*="battle"]').or(page.locator('button:has-text("Battle")')).or(page.locator('text=Battle'));
    
    if (await battleLink.isVisible()) {
      await battleLink.click();
      await page.waitForLoadState('networkidle');
      
      // Look for audio players
      const audioPlayers = page.locator('audio, [class*="audio"], [class*="player"]');
      const playerCount = await audioPlayers.count();
      
      console.log(`Found ${playerCount} audio players`);
      
      if (playerCount > 0) {
        // Test play button interaction
        const playButtons = page.locator('button:has-text("Play"), [aria-label*="play"], .play-button');
        const playButtonCount = await playButtons.count();
        
        if (playButtonCount > 0) {
          await playButtons.first().click();
          await page.waitForTimeout(1000);
          
          // Should not crash after clicking play
          await expect(page.locator('body')).toBeVisible();
        }
      }
    }
  });

  test('should display tournament results', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Look for results or leaderboard
    const resultsElements = page.locator('text=Results, text=Leaderboard, text=Winner, [class*="result"], [class*="leaderboard"]');
    
    if (await resultsElements.first().isVisible()) {
      // Check that results display properly
      await expect(resultsElements.first()).toBeVisible();
      
      // Look for score/ranking information
      const scoreElements = page.locator('text=/\\d+/, [class*="score"], [class*="rank"]');
      await expect(scoreElements.first()).toBeVisible({ timeout: 5000 });
    }
  });
  test('should check for tournament management features', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Navigate to leaderboard or check for saved tournaments
    const leaderboardLink = page.locator('text=LEADERBOARD');
    if (await leaderboardLink.isVisible()) {
      await leaderboardLink.click();
      await page.waitForLoadState('networkidle');
      
      // Look for any existing tournaments or tournament management UI
      const tournaments = page.locator('[class*="tournament"], [data-testid*="tournament"], .tournament-item');
      const tournamentCount = await tournaments.count();
      
      console.log(`Found ${tournamentCount} tournament elements`);
      
      if (tournamentCount > 0) {
        // Look for delete buttons or tournament management options
        const deleteButtons = page.locator('button:has-text("Delete"), button:has-text("Remove"), [aria-label*="delete"], [title*="delete"]');
        const deleteCount = await deleteButtons.count();
        
        console.log(`Found ${deleteCount} delete buttons`);
        
        if (deleteCount === 0) {
          console.log('❌ NO DELETE FUNCTIONALITY FOUND - This is likely the missing feature!');
          // This test will help identify the missing delete functionality
        } else {
          console.log('✅ Delete functionality appears to be available');
        }
      }
    }
  });
});
