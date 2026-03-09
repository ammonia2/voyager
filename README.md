# voyager
MARL Opponent Modeling in Project Malmo, Minecraft

## Project Malmo setup

## Prerequisites

Download and install these before anything else:

- [Miniconda (Windows 64-bit)](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
- [Java 8 JDK (Windows x64 .msi)](https://adoptium.net/temurin/releases/?version=8) — pick the `.msi` installer
- [Malmo 0.37.0 (Windows 64-bit with Boost + Python 3.7)](https://github.com/microsoft/malmo/releases/tag/0.37.0) — download `Malmo-0.37.0-Windows-64bit_withBoost_Python3.7.zip`

### 1. Extract Malmo

Extract the zip to `C:\Malmo`.

### 2. Set JAVA_HOME

Open cmd and run:
```cmd
setx JAVA_HOME "C:\jdk8"
```

Replace `C:\jdk8` with wherever you installed the JDK. Close and reopen cmd after.

### 3. Create Conda Environment
```cmd
conda create -n marl-malmo python=3.7
conda activate marl-malmo
```

### 4. Link Malmo Python Bindings
```cmd
echo C:\Malmo\Python_Examples > C:\Users\<YOUR_USERNAME>\miniconda3\envs\marl-malmo\Lib\site-packages\malmo.pth
```

Replace `<YOUR_USERNAME>` with your Windows username.

Verify it works:
```cmd
python -c "import MalmoPython; print('Malmo OK')"
```

### 5. Build Minecraft Client
```cmd
cd /d C:\Malmo\Minecraft
gradlew.bat setupDecompWorkspace
```

This takes ~5-15 mins on first run. Once done, launch with:
```cmd
gradlew.bat runClient
```

You should see Minecraft 1.11.2 launch with Forge and 5 mods active.

### 6. Install Python Dependencies
```cmd
conda activate marl-malmo
pip install -r requirements.txt
```

---

## Notes

- Always run with `marl-malmo` conda env activated
- `cd` across drives in cmd requires the `/d` flag: `cd /d C:\Malmo\Minecraft`
- The `gradlew.bat setupDecompWorkspace` step only needs to be run once
- Warnings about ForgeGradle version and MC 1.11 mappings during build are harmless