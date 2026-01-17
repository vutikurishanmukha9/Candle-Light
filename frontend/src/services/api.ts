/**
 * API Client Configuration
 * 
 * Centralized API client for communicating with the FastAPI backend.
 * 
 * SECURITY: 
 * - Access token stored in memory only (not localStorage) → XSS-safe
 * - Refresh token stored in HttpOnly cookie by backend → XSS-safe
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// In-memory token storage (XSS-safe - not accessible to injected scripts)
let accessToken: string | null = null;

/**
 * Get stored access token (memory only)
 */
export function getAccessToken(): string | null {
  return accessToken;
}

/**
 * Set access token in memory
 */
export function setAccessToken(token: string | null): void {
  accessToken = token;
}

/**
 * Clear stored tokens (memory + instructs backend to clear cookie on next request)
 */
export function clearTokens(): void {
  accessToken = null;
  // Note: HttpOnly cookie is cleared by backend on logout or failed refresh
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!accessToken;
}

/**
 * API Error class
 */
export class ApiError extends Error {
  status: number;
  details?: Record<string, unknown>;

  constructor(message: string, status: number, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/**
 * Make an API request
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  // Add auth token if available
  if (accessToken) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${accessToken}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
    credentials: 'include',  // IMPORTANT: Send HttpOnly cookies with requests
  });

  // Handle 401 - try to refresh token
  if (response.status === 401) {
    const refreshed = await refreshTokens();
    if (refreshed) {
      // Retry the request with new token
      (headers as Record<string, string>)['Authorization'] = `Bearer ${accessToken}`;
      const retryResponse = await fetch(url, { ...options, headers, credentials: 'include' });
      if (retryResponse.ok) {
        return retryResponse.json();
      }
    }
    // Refresh failed, clear tokens
    clearTokens();
    throw new ApiError('Session expired. Please login again.', 401);
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.error || errorData.detail || 'An error occurred',
      response.status,
      errorData.details
    );
  }

  return response.json();
}

/**
 * Refresh access token using HttpOnly cookie
 * 
 * The refresh_token is automatically sent via HttpOnly cookie.
 * Backend returns new access_token in body and sets new refresh_token cookie.
 */
async function refreshTokens(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',  // Send HttpOnly refresh_token cookie
    });

    if (response.ok) {
      const data = await response.json();
      accessToken = data.access_token;  // Store in memory only
      return true;
    }
    // Log failed refresh for debugging
    console.warn('Token refresh failed:', response.status);
  } catch (error) {
    // Log refresh error for debugging
    console.error('Token refresh error:', error instanceof Error ? error.message : 'Unknown error');
  }
  return false;
}

// ============================================================
// Auth API
// ============================================================

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
}

export const authApi = {
  /**
   * Register a new user
   */
  async register(data: RegisterRequest): Promise<User> {
    return apiRequest<User>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  /**
   * Login with email and password
   * 
   * Access token returned in body (stored in memory).
   * Refresh token set as HttpOnly cookie by backend.
   */
  async login(data: LoginRequest): Promise<AuthTokens> {
    const tokens = await apiRequest<AuthTokens>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    // Store access token in memory only (XSS-safe)
    // Refresh token is automatically set as HttpOnly cookie by backend
    setAccessToken(tokens.access_token);
    return tokens;
  },

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    try {
      await apiRequest('/auth/logout', { method: 'POST' });
    } finally {
      clearTokens();
    }
  },

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<User> {
    return apiRequest<User>('/auth/me');
  },
};

// ============================================================
// Analysis API
// ============================================================

export interface PatternResult {
  name: string;
  type: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  description?: string;
}

export interface EntryTiming {
  signal: 'wait' | 'prepare' | 'trend_continuation' | 'buy_pullback' | 'ready' | 'now';
  timing_description?: string;
  conditions?: string[];
  entry_price_zone?: string;
  stop_loss?: string;
  take_profit?: string;
  risk_reward?: string;
  timeframe?: string;
  scaling_strategy?: string;
}

export interface AnalysisResult {
  id: string;
  image_url: string | null;
  patterns: PatternResult[];
  market_bias: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  reasoning: string | null;
  status: string;
  created_at: string;
  ai_provider: string | null;
  processing_time_ms: number | null;
  entry_timing?: EntryTiming;
}

export interface AnalysisUploadResponse {
  message: string;
  analysis: AnalysisResult;
}

export interface AnalysisListResponse {
  items: AnalysisResult[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export const analysisApi = {
  /**
   * Upload and analyze a chart image
   */
  async uploadChart(file: File): Promise<AnalysisUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const token = getAccessToken();
    const response = await fetch(`${API_BASE_URL}/analysis/upload`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.error || errorData.detail || 'Upload failed',
        response.status
      );
    }

    return response.json();
  },

  /**
   * Get analysis history
   */
  async getHistory(page = 1, pageSize = 10): Promise<AnalysisListResponse> {
    return apiRequest<AnalysisListResponse>(
      `/analysis/history?page=${page}&page_size=${pageSize}`
    );
  },

  /**
   * Get a specific analysis
   */
  async getAnalysis(id: string): Promise<AnalysisResult> {
    return apiRequest<AnalysisResult>(`/analysis/${id}`);
  },

  /**
   * Delete an analysis
   */
  async deleteAnalysis(id: string): Promise<void> {
    await apiRequest(`/analysis/${id}`, { method: 'DELETE' });
  },
};

// ============================================================
// User API
// ============================================================

export interface UserStats {
  total_analyses: number;
  bias_distribution: Record<string, number>;
  member_since: string;
}

export const userApi = {
  /**
   * Get user profile
   */
  async getProfile(): Promise<User> {
    return apiRequest<User>('/users/profile');
  },

  /**
   * Update user profile
   */
  async updateProfile(data: { full_name?: string; avatar_url?: string }): Promise<User> {
    return apiRequest<User>('/users/profile', {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  /**
   * Get user statistics
   */
  async getStats(): Promise<UserStats> {
    return apiRequest<UserStats>('/users/stats');
  },
};

// ============================================================
// Health API
// ============================================================

export const healthApi = {
  /**
   * Check API health
   */
  async check(): Promise<{ status: string; app: string; version: string }> {
    return apiRequest('/health');
  },
};
