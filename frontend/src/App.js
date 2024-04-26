import { Suspense, lazy, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./_styles.scss";

import Spinner from "./components/Spinner";

// Import pages using lazy
const Homepage = lazy(() => import("./pages/Homepage"));
const Error404 = lazy(() => import("./pages/Error404"));

function Content({ setPageTitle, currentPageTitle }) {
  return (
    <Suspense fallback={<Spinner />}>
      <Routes>
        <Route
          path="/"
          element={<Homepage setPageTitle={setPageTitle} currentPageTitle={currentPageTitle} />}
        />
        <Route path="*" element={<Error404 />} />
      </Routes>
    </Suspense>
  );
}

function App() {
  const [pageTitle, setPageTitle] = useState("Brain Lesion Detection and Segmentation System");

  return (
    <Router>
      <Content setPageTitle={setPageTitle} currentPageTitle={pageTitle} />
    </Router>
  );
}

export default App;
