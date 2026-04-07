import { useState } from "react";
import { View, Text, Pressable, StyleSheet, Platform } from "react-native";
import * as DocumentPicker from "expo-document-picker";

interface UploadZoneProps {
  onFileSelected: (file: { uri: string; name: string; size?: number }) => void;
  uploading?: boolean;
  error?: string | null;
}

export default function UploadZone({ onFileSelected, uploading, error }: UploadZoneProps) {
  const [fileName, setFileName] = useState<string | null>(null);

  const pickFile = async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: Platform.OS === "web" ? "audio/midi" : "*/*",
      copyToCacheDirectory: true,
    });

    if (!result.canceled && result.assets?.[0]) {
      const asset = result.assets[0];
      setFileName(asset.name);
      onFileSelected({ uri: asset.uri, name: asset.name, size: asset.size ?? undefined });
    }
  };

  return (
    <View style={styles.container}>
      <Pressable
        style={({ pressed }) => [styles.zone, pressed && styles.zonePressed]}
        onPress={pickFile}
        disabled={uploading}
        testID="upload-zone"
      >
        {uploading ? (
          <>
            <Text style={styles.icon}>...</Text>
            <Text style={styles.label}>Uploading {fileName}...</Text>
          </>
        ) : fileName ? (
          <>
            <Text style={styles.icon}>&#9835;</Text>
            <Text style={styles.label}>{fileName}</Text>
            <Text style={styles.hint}>Tap to change file</Text>
          </>
        ) : (
          <>
            <Text style={styles.icon}>&#8682;</Text>
            <Text style={styles.label}>Tap to select a MIDI file</Text>
            <Text style={styles.hint}>.mid or .midi</Text>
          </>
        )}
      </Pressable>
      {error && <Text style={styles.error}>{error}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginVertical: 16 },
  zone: {
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#2a2a3a",
    borderRadius: 12,
    padding: 40,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#12121a",
  },
  zonePressed: {
    borderColor: "#7c3aed",
    backgroundColor: "rgba(124, 58, 237, 0.05)",
  },
  icon: {
    fontSize: 32,
    color: "#a78bfa",
    marginBottom: 12,
  },
  label: {
    fontSize: 16,
    fontWeight: "500",
    color: "#e8e8f0",
    marginBottom: 4,
  },
  hint: {
    fontSize: 12,
    color: "#7a7a8e",
  },
  error: {
    color: "#f87171",
    fontSize: 13,
    marginTop: 8,
    textAlign: "center",
  },
});
