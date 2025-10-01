"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export default function AuthButton() {
  const router = useRouter();
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    setLoggedIn(!!token);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user_id");
    localStorage.removeItem("username");
    localStorage.removeItem("user_email");
    document.cookie = "access_token=; Max-Age=0; path=/";
    setLoggedIn(false);
    router.push("/auth");
  };

  const handleLogin = () => {
    router.push("/auth");
  };

  return loggedIn ? (
    <button
      onClick={handleLogout}
      className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
    >
      Logout
    </button>
  ) : (
    <button
      onClick={handleLogin}
      className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
    >
      Login
    </button>
  );
}
