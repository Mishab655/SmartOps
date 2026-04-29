import { ChatMessage } from "../types/Message";

interface Props {
  message: ChatMessage;
}

const Message = ({ message }: Props) => {
  return (
    <div style={{ marginBottom: "10px" }}>
      <strong>{message.role === "user" ? "You" : "AI"}:</strong>
      <div>{message.text}</div>

      {message.sql && (
        <details>
          <summary>Show SQL</summary>
          <pre>{message.sql}</pre>
        </details>
      )}
    </div>
  );
};

export default Message;