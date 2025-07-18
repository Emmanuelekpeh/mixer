/**
 * API utility functions for the tournament application
 * Handles API base URL configuration for both development and production environments
 */

// Determine the base URL based on the environment
const getApiBaseUrl = () => {
  return process.env.NODE_ENV === 'production' ? '' : 'http://localhost:10000';
};

/**
 * Make a GET request to the API
 * @param {string} endpoint - The API endpoint to call (e.g., '/api/models')
 * @param {Object} options - Additional fetch options
 * @returns {Promise} - The fetch promise
 */
export const apiGet = async (endpoint, options = {}) => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  });
  
  return response;
};

/**
 * Make a POST request to the API
 * @param {string} endpoint - The API endpoint to call (e.g., '/api/tournaments/create')
 * @param {Object} data - The data to send in the request body
 * @param {Object} options - Additional fetch options
 * @returns {Promise} - The fetch promise
 */
export const apiPost = async (endpoint, data = {}, options = {}) => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    body: JSON.stringify(data),
    ...options
  });
  
  return response;
};

/**
 * Make a PUT request to the API
 * @param {string} endpoint - The API endpoint to call
 * @param {Object} data - The data to send in the request body
 * @param {Object} options - Additional fetch options
 * @returns {Promise} - The fetch promise
 */
export const apiPut = async (endpoint, data, options = {}) => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    body: JSON.stringify(data),
    ...options
  });
  
  return response;
};

/**
 * Upload a file to the API
 * @param {string} endpoint - The API endpoint to call
 * @param {FormData} formData - The form data containing the file
 * @param {Object} options - Additional fetch options
 * @returns {Promise} - The fetch promise
 */
export const apiUpload = async (endpoint, formData, options = {}) => {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  const response = await fetch(url, {
    method: 'POST',
    // Don't set Content-Type header for multipart/form-data
    headers: {
      ...options.headers
    },
    body: formData,
    ...options
  });
  
  return response;
};

export default {
  getApiBaseUrl,
  apiGet,
  apiPost,
  apiPut,
  apiUpload
};