{ mkDerivation, aeson, base, bytestring, lib, linear, text }:
mkDerivation {
  pname = "icfpc2022";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [ aeson base bytestring linear text ];
  executableHaskellDepends = [ aeson base bytestring linear text ];
  license = "unknown";
  mainProgram = "client";
}
