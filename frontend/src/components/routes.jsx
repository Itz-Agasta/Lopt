import { Component } from "react";
import { createBrowserRouter } from "react-router";
import Home from "../pages/Home";
import Dashboard from "../pages/Dashboard";
import Results from "../pages/Results";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/home",
    element: <Home />,
  },
  {
    path: "/dashboard",
    element: <Dashboard />,
  },
  {
    path: "/results",
    element: <Results />,
  },
]);

export default router;
