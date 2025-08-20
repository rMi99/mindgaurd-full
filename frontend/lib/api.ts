import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// API Configuration
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/api';
const API_TIMEOUT = 30000; // 30 seconds

function hardLogout() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('mindguard_token');
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    document.cookie = 'access_token=; Max-Age=0; path=/';
    document.cookie = 'refresh_token=; Max-Age=0; path=/';
    window.location.href = '/auth/login';
  }
}

// Types
export interface User {
  id: string;
  email: string;
  full_name: string;
  age?: number;
  gender?: string;
  created_at: string;
  last_login?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  full_name: string;
  age?: number;
  gender?: string;
}

export interface AssessmentSubmission {
  fullName: string;
  age: number;
  gender?: string;
  occupation?: string;
  email: string;
  sleepHours: number;
  exerciseFrequency: number;
  dietQuality: number;
  stressLevel: number;
  moodScore: number;
  mentalHealthHistory?: string[];
  socialConnections: number;
  workLifeBalance: number;
  financialStress: number;
  additionalNotes?: string;
}

export interface AssessmentResult {
  assessment_id: string;
  risk_level: 'low' | 'normal' | 'high';
  confidence: number;
  health_score: number;
  recommendations: string[];
  brain_heal_activities: Array<{
    name: string;
    duration: string;
    description: string;
    steps: string[];
    benefits: string[];
    difficulty: string;
  }>;
  weekly_plan: Record<string, string[]>;
  created_at: string;
}

export interface DashboardData {
  user: User;
  recent_assessments: AssessmentResult[];
  health_trends: Array<{
    date: string;
    sleep_hours: number;
    stress_level: number;
    exercise_frequency: number;
    mood_score: number;
    energy_level: number;
    social_connections: number;
    diet_quality: number;
  }>;
  current_risk_level: 'low' | 'normal' | 'high';
  overall_health_score: number;
  weekly_challenge: {
    title: string;
    description: string;
    daily_challenges: string[];
    goal: string;
  };
}

export interface ApiError {
  message: string;
  status: number;
  details?: any;
}

// API Client Class
class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Initialize token from localStorage
    this.token = this.getStoredToken();
    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const localToken = this.getStoredToken();
        if (localToken) {
          this.token = localToken;
        }
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor to handle token refresh and errors
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      async (error: AxiosError) => {
        const status = error.response?.status;
        if (status === 401) {
          // Force logout on unauthorized; do NOT loop refresh
          hardLogout();
          return Promise.reject(this.handleError(error));
        }
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private getStoredToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('mindguard_token') || localStorage.getItem('access_token');
    }
    return null;
  }

  private setStoredToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('mindguard_token', token);
      localStorage.setItem('access_token', token);
    }
  }

  private clearStoredToken(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('mindguard_token');
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      document.cookie = 'access_token=; Max-Age=0; path=/';
      document.cookie = 'refresh_token=; Max-Age=0; path=/';
    }
  }

  private setToken(token: string): void {
    this.token = token;
    this.setStoredToken(token);
  }

  private clearToken(): void {
    this.token = null;
    this.clearStoredToken();
  }

  private async refreshToken(): Promise<string | null> {
    // Disabled to avoid loops; force logout on 401 instead
    return null;
  }

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      return {
        message: (error.response.data as any)?.detail || (error.response.data as any)?.message || 'An error occurred',
        status: error.response.status!,
        details: error.response.data,
      };
    } else if (error.request) {
      return {
        message: 'No response from server. Please check your internet connection.',
        status: 0,
        details: error.request,
      };
    } else {
      return {
        message: error.message || 'An unexpected error occurred',
        status: 0,
        details: error,
      };
    }
  }

  // Authentication Methods
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/login', credentials);
    this.setToken(response.data.access_token);
    return response.data;
  }

  async register(data: RegisterData): Promise<AuthResponse> {
    const response = await this.client.post<AuthResponse>('/auth/register', data);
    this.setToken(response.data.access_token);
    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.clearToken();
      if (typeof window !== 'undefined') window.location.href = '/auth/login';
    }
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  async updateProfile(profileData: Partial<User>): Promise<User> {
    const response = await this.client.put<User>('/auth/profile', profileData);
    return response.data;
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<{ message: string }> {
    const response = await this.client.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
    return response.data;
  }

  // Assessment Methods
  async submitAssessment(assessmentData: AssessmentSubmission): Promise<AssessmentResult> {
    const response = await this.client.post<AssessmentResult>('/assessment/submit', assessmentData);
    return response.data;
  }

  async getAssessmentHistory(limit: number = 10): Promise<any[]> {
    const response = await this.client.get<any[]>(`/assessment/history?limit=${limit}`);
    return response.data as any[];
  }

  // Dashboard Methods
  async getDashboardData(): Promise<DashboardData> {
    const response = await this.client.get<DashboardData>('/dashboard/data');
    return response.data;
  }

  async getHealthTrends(timeRange: '7d' | '30d' | '90d' | '1y' = '30d') {
    const response = await this.client.get(`/dashboard/trends?range=${timeRange}`);
    return response.data;
  }

  // Recommendations Methods
  async getRecommendations(riskLevel: 'low' | 'normal' | 'high'): Promise<any> {
    const response = await this.client.get(`/recommendations?risk_level=${riskLevel}`);
    return response.data;
  }

  // Utility Methods
  isAuthenticated(): boolean {
    return !!this.getStoredToken();
  }

  getToken(): string | null {
    return this.getStoredToken();
  }

  // Health Check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

// Create and export singleton instance
export const apiClient = new ApiClient();

// Export convenience functions
export const login = (credentials: LoginCredentials) => apiClient.login(credentials);
export const register = (data: RegisterData) => apiClient.register(data);
export const logout = () => apiClient.logout();
export const getCurrentUser = () => apiClient.getCurrentUser();
export const submitAssessment = (data: AssessmentSubmission) => apiClient.submitAssessment(data);
export const getDashboardData = () => apiClient.getDashboardData();
export const getAssessmentHistory = (limit?: number) => apiClient.getAssessmentHistory(limit);
export const getRecommendations = (riskLevel: 'low' | 'normal' | 'high') => apiClient.getRecommendations(riskLevel);
export const isAuthenticated = () => apiClient.isAuthenticated();

// Export types
export type {
  User,
  AuthResponse,
  LoginCredentials,
  RegisterData,
  AssessmentSubmission,
  AssessmentResult,
  DashboardData,
  ApiError,
};

// Export client instance for advanced usage
export default apiClient; 