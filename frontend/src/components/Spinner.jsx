import React from "react";
import "../_styles.scss";

const spinnerStyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  height: "100vh"
};

const spinnerInnerStyle = {
  width: "50px",
  height: "50px",
  border: "6px solid rgba(0,0,0,.1)",
  // FIXTHIS change the color to $color4
  borderLeftColor: "#115038",
  borderRadius: "50%",
  animation: "spin 1s linear infinite"
};

const Spinner = () => (
  <div style={spinnerStyle}>
    <div style={spinnerInnerStyle}></div>
  </div>
);

export default Spinner;
