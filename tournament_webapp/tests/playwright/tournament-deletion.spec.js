// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Test for checking tournament deletion functionality
 */
test.describe('Tournament Deletion Tests', () => {
  test('should delete a tournament when delete button is clicked', async ({ page }) => {
    // Navigate to the saved tournaments page
    await page.goto('http://localhost:3000/saved');
    
    // Wait for the page to load
    await page.waitForSelector('h2:has-text("Your Saved Tournaments")', { timeout: 10000 });
    
    // Check if there are any tournaments
    const hasTournaments = await page.isVisible('.tournament-card');
    
    if (hasTournaments) {
      // Get the number of tournaments before deletion
      const beforeCount = await page.locator('.tournament-card').count();
      
      // Click the delete button on the first tournament
      await page.locator('.tournament-card').first().locator('button:has-text("Delete")').click();
      
      // Accept the confirmation dialog
      await page.locator('text=Are you sure').waitFor({ state: 'visible', timeout: 5000 });
      await page.keyboard.press('Enter');
      
      // Wait for deletion and UI update
      await page.waitForTimeout(1000);
      
      // Check if the number of tournaments has decreased
      const afterCount = await page.locator('.tournament-card').count();
      expect(afterCount).toBe(beforeCount - 1);
      
      // Check for success toast
      await expect(page.locator('text=Tournament deleted successfully')).toBeVisible();
    } else {
      test.skip('No tournaments available to delete');
    }
  });

  test('API endpoint should return 200 for tournament deletion', async ({ request }) => {
    // First get a list of tournaments
    const response = await request.get('http://localhost:10000/api/tournaments');
    const data = await response.json();
    
    if (data.tournaments && data.tournaments.length > 0) {
      const tournamentId = data.tournaments[0].id || data.tournaments[0].tournament_id;
      
      // Try to delete the tournament
      const deleteResponse = await request.delete(`http://localhost:10000/api/tournaments/${tournamentId}`, {
        data: { user_id: 'test_user' }
      });
      
      expect(deleteResponse.status()).toBe(200);
      
      const responseData = await deleteResponse.json();
      expect(responseData.success).toBe(true);
    } else {
      test.skip('No tournaments available to test deletion');
    }
  });
});
