import ReactDOM from "react-dom/client";
import App from "./App";
import { StrictMode } from "react";
// import reportWebVitals from "./reportWebVitals";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <StrictMode>
    <App />
  </StrictMode>
);

// Log web vitals to the console
// reportWebVitals(console.log);
