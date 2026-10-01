// A small fetch-based SSE client that works with POST /answer in any browser UI.
export async function streamAnswer(question, onEvent, { endpoint = "/answer", signal } = {}) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok) {
    throw new Error(`Answer request failed (${response.status})`);
  }
  if (!response.body) {
    throw new Error("This browser cannot read the answer stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseEvent(frame);
        if (event) {
          onEvent(event);
          if (event.type === "done") return event.data;
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
    throw new Error("Answer stream ended before the final event");
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}

function parseEvent(frame) {
  let type = "message";
  const data = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return data.length ? { type, data: JSON.parse(data.join("\n")) } : null;
}
