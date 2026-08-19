import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PawGPT, find the dog that fits your life",
  description:
    "Describe your home, your hours and your patience, and get dog breeds that " +
    "genuinely match, with the numbers behind every recommendation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Satoshi is served from Fontshare rather than bundled. Opening the
            connections early means the TLS handshake is not still in progress when
            the stylesheet asks for the font files. `display=swap` is set on the
            request, so text paints immediately in the fallback and reflows once
            Satoshi arrives rather than sitting invisible. */}
        <link rel="preconnect" href="https://api.fontshare.com" />
        <link rel="preconnect" href="https://cdn.fontshare.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700,900&display=swap"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
