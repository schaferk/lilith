import { useQuery } from "@tanstack/react-query";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Station {
  station_id: string;
  name: string;
  state: string;
  country: string;
  latitude: number;
  longitude: number;
  elevation: number;
  current_temp: number;
  forecast_high: number;
  forecast_low: number;
  actual_high: number;
  actual_low: number;
  trend: "improving" | "stable" | "declining";
  last_observation: string;
  temp_error_avg: number;
  precip_accuracy: number;
  high_error?: number;
  low_error?: number;
  start_date?: string;
  end_date?: string;
}

interface StationListResponse {
  stations: Station[];
  total: number;
  page: number;
  page_size: number;
}

interface UseStationsParams {
  page?: number;
  pageSize?: number;
  latitude?: number;
  longitude?: number;
  radius?: number;
}

async function fetchStations(params: UseStationsParams): Promise<StationListResponse> {
  const queryParams = new URLSearchParams();
  if (params.page) queryParams.append("page", params.page.toString());
  if (params.pageSize) queryParams.append("page_size", params.pageSize.toString());
  if (params.latitude) queryParams.append("latitude", params.latitude.toString());
  if (params.longitude) queryParams.append("longitude", params.longitude.toString());
  if (params.radius) queryParams.append("radius", params.radius.toString());

  const response = await axios.get<StationListResponse>(`${API_URL}/v1/stations?${queryParams}`);
  return response.data;
}

export function useStations(params: UseStationsParams = {}) {
  return useQuery({
    queryKey: ["stations", params],
    queryFn: () => fetchStations(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    placeholderData: (previousData) => previousData, // Keep previous data while fetching new page
  });
}
