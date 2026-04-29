import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

export interface ApiResponse {
  answer: string;
  sql?: string;
}

export const sendMessage = async (
  question: string
): Promise<ApiResponse> => {
  const response = await API.post("/chat", {
    question,
  });
  return response.data;
};