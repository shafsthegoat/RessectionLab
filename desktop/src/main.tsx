import "./tokens.css";
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./studio.css";
import "./study-inspector.css";
import "./library.css";
import "./viewer/viewport-studio.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
