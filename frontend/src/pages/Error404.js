import React, { useEffect } from "react";
import "../_styles.scss";

import { Link } from "react-router-dom";

function Error404() {

  useEffect(() => {
    document.title = "Page Not Found";
  }, []);

  return (
    <div className="p-0 m-0 w-100 justify-content-center align-items-center">
      <div className="p-0 px-4 px-lg-0 m-0 my-5 text-center">
        <div className="p-0 m-0 mx-auto text-start col-12 col-lg-5">
          <h2>Oops! It looks like the page you were trying to access couldn't be found.</h2>
          <br />
          <p>We're sorry for the inconvenience. Please try the following:</p>
          <ul>
            <li>Check the URL to ensure it's spelled correctly</li>
            <li>Use our navbar to find what you're looking for</li>
            <li>Visit our homepage to explore more of our website</li>
          </ul>
          <p>If you're still having trouble, please <a href="#" target="_blank" rel="noopener noreferrer" className="color4">contact our support team</a> for assistance.</p>
          <p>Thank you for your patience and understanding. We hope you enjoy browsing the rest of our website!</p>
          <br />
        </div>
        <Link to="/" className="btn btn1">RETURN TO HOMEPAGE</Link>
      </div>
      <div className="p-0 m-0 the-border-1" />
    </div>
  );
}

export default Error404;
