import { v4 as uuidv4 } from 'uuid';

const ANON_USER_KEY = 'mt_anon_user_id';
const AUTH_USER_KEY = 'mt_auth_user_id';

/**
 * Get or create an anonymous user ID for local storage
 * Returns the stored ID or generates a new one if none exists
 */
export function getAnonymousUserId(): string {
  let anonId = localStorage.getItem(ANON_USER_KEY);
  
  if (!anonId) {
    anonId = uuidv4();
    localStorage.setItem(ANON_USER_KEY, anonId);
    console.log('Generated new anonymous user ID:', anonId);
  }
  
  return anonId;
}

/**
 * Set the authenticated user ID after login
 */
export function setUserId(userId: string): void {
  localStorage.setItem(AUTH_USER_KEY, userId);
}

/**
 * Get the currently active user ID:
 * - Returns authenticated user ID if logged in
 * - Returns anonymous user ID if logged out
 */
export function getUserId(): string {
  const authId = localStorage.getItem(AUTH_USER_KEY);
  if (authId) {
    return authId;
  }
  return getAnonymousUserId();
}

/**
 * Clear authenticated user ID on logout (keeps anonymous ID intact)
 */
export function clearAuthUserId(): void {
  localStorage.removeItem(AUTH_USER_KEY);
}
