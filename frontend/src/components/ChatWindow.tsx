import { useState } from "react";
import { ChatMessage } from "../types/Message";
import { sendMessage } from "../services/api";
import Message from "./Message";
import InputBox from "./InputBox";

const ChatWindow = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSend = async (input: string) => {
    const userMessage: ChatMessage = {
      role: "user",
      text: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const data = await sendMessage(input);

      const botMessage: ChatMessage = {
        role: "bot",
        text: data.answer || "No answer provided",
        sql: data.sql,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Error connecting to backend" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: "600px", margin: "auto" }}>
      <h2>AI Business Advisor</h2>

      <div style={{ minHeight: "400px" }}>
        {messages.map((msg, index) => (
          <Message key={index} message={msg} />
        ))}

        {loading && <p>Thinking...</p>}
      </div>

      <InputBox onSend={handleSend} />
    </div>
  );
};

export default ChatWindow;