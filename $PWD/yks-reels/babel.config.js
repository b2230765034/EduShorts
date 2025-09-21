// babel.config.js
module.exports = function (api) {
    api.cache(true);
    return {
      presets: [
        'babel-preset-expo',
        ['@babel/preset-react', { runtime: 'automatic' }], // <-- modern JSX
      ],
      plugins: [
        'expo-router/babel',                // app/ klasörü kullanıyorsun
        'react-native-reanimated/plugin',   // <-- EN SONDA KALSIN
      ],
    };
  };
  