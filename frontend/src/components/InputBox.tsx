import { useState } from "react";

interface Props {
  onSend: (input: string) => void;
}

const InputBox = ({ onSend }: Props) => {
  const [input, setInput] = useState<string>("");

  const handleSubmit = () => {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <div style={{ display: "flex", marginTop: "10px" }}>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask about sales, forecast, churn..."
        style={{ flex: 1 }}
      />
      <button onClick={handleSubmit}>Send</button>
    </div>
  );
};

export default InputBox;