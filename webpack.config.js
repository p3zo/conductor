const path = require('path');

const HtmlWebpackPlugin = require('html-webpack-plugin');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = (env, argv) => {
    // The mode comes from the --mode flag in the package.json scripts, so the
    // deployed build can't silently fall back to an unminified development one.
    const production = argv.mode === 'production';

    const config = {
        mode: production ? 'production' : 'development',
        entry: {
            main: './src/index.js',
        },
        output: {
            path: path.resolve(__dirname, 'dist'),
            clean: true,
            filename: 'bundle.js',
        },
        // An inline source map is 16 MB of base64 in the bundle; keep it out of the deployed build.
        devtool: production ? false : 'inline-source-map',
        devServer: {
            headers: {
                // These two headers are required for cross origin isolation.
                'Cross-Origin-Opener-Policy': 'same-origin',
                'Cross-Origin-Embedder-Policy': 'require-corp'
            },
            https: true,
            host: '0.0.0.0',
            port: 8090,
            historyApiFallback: {
                disableDotRule: true,
            },
            watchFiles: ['src/**/*'],
        },
        resolve: {
            extensions: ['.tsx', '.ts', '.js'],
        },
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
            ],
        },
        plugins: [
            new HtmlWebpackPlugin({
                template: 'index.html',
                filename: 'index.html',
            }),
            new CopyWebpackPlugin({
                patterns: [
                    {from: 'node_modules/@tensorflow/tfjs-backend-wasm/dist/*.wasm', to: 'wasm/[name][ext]'},
                    {from: 'node_modules/@handtracking.io/yoha/models/', to: './'},
                    {from: 'static', to: './'},
                ]
            })
        ],
        optimization: {
            minimizer: [new TerserPlugin({
                extractComments: false,
                exclude: /\.min\./,
            })],
        },
    };

    return [config];
}
